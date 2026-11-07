/* SPDX-FileCopyrightText: 2026 Tesseract Studio
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * akar-render: the photoreal renderer for Akar (by Tesseract Studio), built on Cycles.
 *
 * Akar writes the model to render as one scene file (see docs/scene-format.md) and starts this
 * program in its own process, so a crash here never takes Akar down. It renders the scene with
 * Cycles on the best device there is (OptiX on NVIDIA RTX cards, then CUDA, then the
 * processor), writes the picture as it improves and when it's done, and reports progress on
 * standard output, one line at a time:
 *
 *   DEVICE <type> <name>
 *   PROGRESS <0..1> <samples done> <samples asked> <seconds so far> <seconds left, -1 unknown>
 *   EXPOSURE <e> <middle> <seconds>  (1.0.5: the exposure rendered with, from the model's
 *                           middle brightness in a first small picture; the files hold the light
 *                           without it)
 *   WHITE <r> <g> <b>       (1.0.7: the colour of the light falling on the model in that first
 *                           picture, its brightness 1: the sky's blue on a façade in shade, a
 *                           lamp's warmth; Akar balances the picture's colour by it)
 *   PREVIEW <path>          (a newer picture is there)
 *   DONE <path> <seconds>
 *   ERROR <message>
 *
 * A line "stop" on standard input ends the render early, keeping what's done.
 * The pictures are 32-bit float RGBA, linear (scene-referred), top row first, as a small header
 * ("AKRI", width, height, samples) and the pixels (RGB light; A: 1 where the model is, 0 where
 * the sky is seen); Akar applies exposure and the tone curve.
 */

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "device/device.h"
#include "scene/background.h"
#include "scene/camera.h"
#include "scene/film.h"
#include "scene/integrator.h"
#include "scene/light.h"
#include "scene/mesh.h"
#include "scene/object.h"
#include "scene/pass.h"
#include "scene/scene.h"
#include "scene/shader.h"
#include "scene/shader_graph.h"
#include "scene/shader_nodes.h"
#include "session/buffers.h"
#include "session/output_driver.h"
#include "session/session.h"
#include "util/colorspace.h"
#include "util/log.h"
#include "util/path.h"
#include "util/progress.h"
#include "util/string.h"
#include "util/system.h"
#include "util/transform.h"
#include "util/unique_ptr.h"
#include "util/version.h"
#include "util/xml.h"

CCL_NAMESPACE_BEGIN

static std::mutex print_mutex;

static void say(const std::string &line)
{
  std::lock_guard<std::mutex> lock(print_mutex);
  std::fputs(line.c_str(), stdout);
  std::fputc('\n', stdout);
  std::fflush(stdout);
}

/* Ends the program at once. (Not exit(): the thread waiting for "stop" holds standard input,
 * and exit()'s tidying up of the standard streams would wait for it for ever.) */
[[noreturn]] static void quit(const int code)
{
  std::fflush(stdout);
  std::fflush(stderr);
  std::_Exit(code);
}

[[noreturn]] static void fail(const std::string &message)
{
  say("ERROR " + message);
  quit(1);
}

/* ---------------------------------------------------------------------------------------- */
/* Reading the scene file                                                                    */
/* ---------------------------------------------------------------------------------------- */

static std::vector<float> floats(const char *text)
{
  std::vector<float> out;
  const char *p = text ? text : "";
  while (*p) {
    char *end = nullptr;
    const float v = std::strtof(p, &end);
    if (end == p) {
      ++p;
      continue;
    }
    out.push_back(v);
    p = end;
  }
  return out;
}

static float3 f3(const xml_node &n, const char *name, const float3 fallback)
{
  const std::vector<float> v = floats(n.attribute(name).value());
  return v.size() >= 3 ? make_float3(v[0], v[1], v[2]) : fallback;
}

static float f1(const xml_node &n, const char *name, const float fallback)
{
  const xml_attribute a = n.attribute(name);
  return a ? a.as_float(fallback) : fallback;
}

static int i1(const xml_node &n, const char *name, const int fallback)
{
  const xml_attribute a = n.attribute(name);
  return a ? a.as_int(fallback) : fallback;
}

/* A transform from its 12 numbers, three rows of four (rotation and scale, then the move). */
static Transform tfm12(const xml_node &n, const char *name)
{
  const std::vector<float> m = floats(n.attribute(name).value());
  if (m.size() < 12) {
    return transform_identity();
  }
  Transform t;
  t.x = make_float4(m[0], m[1], m[2], m[3]);
  t.y = make_float4(m[4], m[5], m[6], m[7]);
  t.z = make_float4(m[8], m[9], m[10], m[11]);
  return t;
}

/* A frame whose -Z points along `dir` (where lights shine), placed at `pos`; its X as near
 * `along` as it can be (an area light's length), when that's given. */
static Transform facing(const float3 pos, float3 dir, const float3 along = zero_float3())
{
  dir = safe_normalize(dir);
  if (len_squared(dir) < 1e-12f) {
    dir = make_float3(0.0f, 0.0f, -1.0f);
  }
  const float3 z = -dir;
  float3 x = along - z * dot(along, z);
  if (len_squared(x) < 1e-8f) {
    const float3 helper = fabsf(z.z) < 0.9f ? make_float3(0.0f, 0.0f, 1.0f) : make_float3(1.0f, 0.0f, 0.0f);
    x = cross(helper, z);
  }
  x = safe_normalize(x);
  const float3 y = cross(z, x);
  Transform t;
  t.x = make_float4(x.x, y.x, z.x, pos.x);
  t.y = make_float4(x.y, y.y, z.y, pos.y);
  t.z = make_float4(x.z, y.z, z.z, pos.z);
  return t;
}

struct SceneFile {
  xml_document doc;
  std::vector<char> blob;
  std::string folder;
};

static void read_scene_file(const std::string &path, SceneFile &file)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    fail("can't open the scene file " + path);
  }
  char magic[4];
  uint32_t version = 0, xml_length = 0;
  in.read(magic, 4);
  in.read(reinterpret_cast<char *>(&version), 4);
  in.read(reinterpret_cast<char *>(&xml_length), 4);
  if (!in || std::memcmp(magic, "AKRS", 4) != 0 || version != 1) {
    fail("not an Akar render scene (or a newer one): " + path);
  }
  std::string xml(xml_length, '\0');
  in.read(xml.data(), xml_length);
  const std::streamoff here = in.tellg();
  const std::streamoff blob_start = (here + 15) & ~std::streamoff(15);
  in.seekg(0, std::ios::end);
  const std::streamoff end = in.tellg();
  if (end < blob_start) {
    fail("the scene file is cut short");
  }
  file.blob.resize(size_t(end - blob_start));
  in.seekg(blob_start);
  in.read(file.blob.data(), std::streamsize(file.blob.size()));
  const xml_parse_result ok = file.doc.load_string(xml.c_str());
  if (!ok) {
    fail(std::string("the scene's description can't be read: ") + ok.description());
  }
  file.folder = path_dirname(path);
}

template<typename T> static const T *blob_at(const SceneFile &file, const size_t offset, const size_t count)
{
  if (offset + count * sizeof(T) > file.blob.size()) {
    fail("the scene file's data is cut short");
  }
  return reinterpret_cast<const T *>(file.blob.data() + offset);
}

/* ---------------------------------------------------------------------------------------- */
/* Building the Cycles scene                                                                 */
/* ---------------------------------------------------------------------------------------- */

struct Built {
  std::vector<Shader *> materials;
  std::vector<Mesh *> meshes;
  int width = 1280, height = 720;
};

/* One material's surface, added to a graph: its BSDF's output. <uv_map> names the texture
 * coordinates its pictures use (null: the mesh's own, "UVMap"). */
static ShaderOutput *add_surface(ShaderGraph *graph, const SceneFile &file, const xml_node &m,
                                 const std::vector<std::pair<std::string, std::string>> &textures,
                                 const char *uv_map)
{
  PrincipledBsdfNode *bsdf = graph->create_node<PrincipledBsdfNode>();
  bsdf->set_base_color(f3(m, "color", make_float3(0.8f, 0.8f, 0.8f)));
  bsdf->set_roughness(f1(m, "roughness", 0.6f));
  bsdf->set_metallic(f1(m, "metallic", 0.0f));
  bsdf->set_ior(f1(m, "ior", 1.5f));
  bsdf->set_specular_ior_level(f1(m, "specular", 0.5f));
  bsdf->set_transmission_weight(f1(m, "transmission", 0.0f));
  bsdf->set_alpha(f1(m, "alpha", 1.0f));
  const float emission = f1(m, "emission_strength", 0.0f);
  if (emission > 0.0f) {
    bsdf->set_emission_color(f3(m, "emission", make_float3(1.0f, 1.0f, 1.0f)));
    bsdf->set_emission_strength(emission);
  }

  auto texture_node = [&](const int index, const bool colour) -> ImageTextureNode * {
    if (index < 0 || index >= int(textures.size())) {
      return nullptr;
    }
    ImageTextureNode *tex = graph->create_node<ImageTextureNode>();
    tex->set_filename(ustring(path_join(file.folder, textures[size_t(index)].first)));
    /* (Colour pictures are sRGB. Cycles' "__builtin_srgb" is only made linear through OpenColorIO,
     * which the pack doesn't carry, so they'd be read as if linear: pale and greyed. Scene linear
     * with the sRGB curve is kept as it is and made linear as it's looked up, with no OpenColorIO.) */
    tex->set_colorspace(colour ? u_colorspace_scene_linear_srgb : u_colorspace_data);
    tex->set_interpolation(INTERPOLATION_LINEAR);
    tex->set_extension(EXTENSION_REPEAT);
    tex->set_alpha_type(IMAGE_ALPHA_AUTO);
    if (uv_map != nullptr) {
      UVMapNode *uv = graph->create_node<UVMapNode>();
      uv->set_attribute(ustring(uv_map));
      graph->connect(uv->output("UV"), tex->input("Vector"));
    }
    return tex;
  };

  if (ImageTextureNode *tex = texture_node(i1(m, "texture", -1), true)) {
    if (i1(m, "tint", 0)) {
      /* A colour multiplied over the picture (Akar's tinted textures). */
      VectorMathNode *mul = graph->create_node<VectorMathNode>();
      mul->set_math_type(NODE_VECTOR_MATH_MULTIPLY);
      mul->set_vector2(f3(m, "tint_color", one_float3()));
      graph->connect(tex->output("Color"), mul->input("Vector1"));
      graph->connect(mul->output("Vector"), bsdf->input("Base Color"));
    }
    else {
      graph->connect(tex->output("Color"), bsdf->input("Base Color"));
    }
    if (i1(m, "cutout", 0)) {
      const float alpha = f1(m, "alpha", 1.0f);
      if (alpha < 0.999f) {
        /* 1.0.7: a cut-out picture on a material that's see-through itself (leaves at 88 %):
         * the picture's alpha times the material's. */
        MathNode *times = graph->create_node<MathNode>();
        times->set_math_type(NODE_MATH_MULTIPLY);
        times->set_value2(alpha);
        graph->connect(tex->output("Alpha"), times->input("Value1"));
        graph->connect(times->output("Value"), bsdf->input("Alpha"));
      }
      else {
        graph->connect(tex->output("Alpha"), bsdf->input("Alpha"));
      }
    }
  }
  if (ImageTextureNode *rough = texture_node(i1(m, "roughness_map", -1), false)) {
    SeparateColorNode *split = graph->create_node<SeparateColorNode>();
    graph->connect(rough->output("Color"), split->input("Color"));
    graph->connect(split->output("Green"), bsdf->input("Roughness")); /* (glTF: G roughness, B metal) */
    if (i1(m, "metallic_from_map", 0)) {
      graph->connect(split->output("Blue"), bsdf->input("Metallic"));
    }
  }
  NormalMapNode *bumps = nullptr;
  if (ImageTextureNode *normal = texture_node(i1(m, "normal_map", -1), false)) {
    NormalMapNode *nm = graph->create_node<NormalMapNode>();
    nm->set_space(NODE_NORMAL_MAP_TANGENT);
    nm->set_strength(f1(m, "normal_strength", 1.0f));
    if (uv_map != nullptr) {
      nm->set_attribute(ustring(uv_map));
    }
    graph->connect(normal->output("Color"), nm->input("Color"));
    graph->connect(nm->output("Normal"), bsdf->input("Normal"));
    bumps = nm;
  }

  if (i1(m, "glass", 0)) {
    /* 1.0.8: see-through glass (Akar: a plain colour with opacity under 100 %). Before, its
     * opacity mixed the surface with "nothing there", so its reflections were weakened as much as
     * its colour: a wine glass at 20 % kept a fifth of an already faint reflection and showed as
     * grey film. Now the see-through part is clear (tinted by the colour, as much as it's glossy)
     * with its own reflection on top, as strong as the angle and the reflectivity make it
     * (Fresnel; reflectivity 50 % = glass's own, IOR 1.5), and sharp or soft as it's glossy. A
     * matt one keeps a body of its colour (frosted). It's transparent, not refracting, so sunlight
     * still comes through a window and a single pane isn't bent. */
    const float alpha = clamp(f1(m, "alpha", 1.0f), 0.0f, 1.0f);
    const float rough = clamp(f1(m, "roughness", 0.03f), 0.0f, 1.0f);
    const float gloss = 1.0f - rough;
    const float3 colour = f3(m, "color", make_float3(0.8f, 0.8f, 0.8f));
    bsdf->set_alpha(1.0f);
    TransparentBsdfNode *clear = graph->create_node<TransparentBsdfNode>();
    /* (The colour and opacity describe a pane as it's seen in SketchUp or Akar's view; a pane is
     * usually two faces, so each face takes the square root of the pane's tint, and through its
     * two faces it's the pane's own.) */
    const float3 pane = one_float3() + (colour - one_float3()) * (alpha * gloss);
    clear->set_color(make_float3(sqrtf(fmaxf(pane.x, 0.0f)), sqrtf(fmaxf(pane.y, 0.0f)), sqrtf(fmaxf(pane.z, 0.0f))));
    GlossyBsdfNode *shine = graph->create_node<GlossyBsdfNode>();
    shine->set_color(one_float3());
    shine->set_roughness(rough);
    const float ior = 1.0f + clamp(f1(m, "specular", 0.5f), 0.0f, 1.0f);
    FresnelNode *fresnel = graph->create_node<FresnelNode>();
    /* (Cycles' Fresnel takes a back face for glass to air: past about 42 degrees all of it
     * reflects. As this glass doesn't bend the light, a ray through a pane seen at a slant then
     * stayed inside it, bouncing between its two faces until it ran out of bounces: a window
     * seen from the side came out black. A thin pane is air to glass on both sides, so a back
     * face is given 1/IOR, which Cycles turns back into the IOR.) */
    GeometryNode *geometry = graph->create_node<GeometryNode>();
    MathNode *side = graph->create_node<MathNode>();
    side->set_math_type(NODE_MATH_MULTIPLY_ADD);
    side->set_value2(1.0f / ior - ior);
    side->set_value3(ior);
    graph->connect(geometry->output("Backfacing"), side->input("Value1"));
    graph->connect(side->output("Value"), fresnel->input("IOR"));
    if (bumps != nullptr) {
      graph->connect(bumps->output("Normal"), shine->input("Normal"));
      graph->connect(bumps->output("Normal"), fresnel->input("Normal"));
    }
    MixClosureNode *see = graph->create_node<MixClosureNode>();
    graph->connect(fresnel->output("Fac"), see->input("Fac"));
    graph->connect(clear->output("BSDF"), see->input("Closure1"));
    graph->connect(shine->output("BSDF"), see->input("Closure2"));
    MixClosureNode *glass = graph->create_node<MixClosureNode>();
    glass->set_fac(alpha * rough); /* (the body: none for clear glass, its colour for frosted) */
    graph->connect(see->output("Closure"), glass->input("Closure1"));
    graph->connect(bsdf->output("BSDF"), glass->input("Closure2"));
    return glass->output("Closure");
  }

  return bsdf->output("BSDF");
}

/* A material, or a pair: front="i" back="j" names two earlier materials, the front's seen from
 * the front of a face and the back's from behind it (the back's pictures on the mesh's "UVBack"). */
static Shader *make_material(Scene *scene, const SceneFile &file, const xml_node &m,
                             const std::vector<std::pair<std::string, std::string>> &textures,
                             const std::vector<xml_node> &nodes)
{
  Shader *shader = scene->create_node<Shader>();
  shader->name = ustring(m.attribute("name").value());
  unique_ptr<ShaderGraph> graph = make_unique<ShaderGraph>();
  const int front = i1(m, "front", -1), back = i1(m, "back", -1);
  ShaderOutput *surface = nullptr;
  if (front >= 0 && back >= 0 && front < int(nodes.size()) && back < int(nodes.size())) {
    ShaderOutput *a = add_surface(graph.get(), file, nodes[size_t(front)], textures, nullptr);
    ShaderOutput *b = add_surface(graph.get(), file, nodes[size_t(back)], textures, "UVBack");
    GeometryNode *geometry = graph->create_node<GeometryNode>();
    MixClosureNode *mix = graph->create_node<MixClosureNode>();
    graph->connect(geometry->output("Backfacing"), mix->input("Fac"));
    graph->connect(a, mix->input("Closure1"));
    graph->connect(b, mix->input("Closure2"));
    surface = mix->output("Closure");
  }
  else {
    surface = add_surface(graph.get(), file, m, textures, nullptr);
  }
  graph->connect(surface, graph->output()->input("Surface"));
  shader->set_graph(std::move(graph));
  shader->tag_update(scene);
  return shader;
}

static Mesh *make_mesh(Scene *scene, const SceneFile &file, const xml_node &n, const Built &built)
{
  const size_t nv = size_t(i1(n, "vertices", 0));
  const size_t nt = size_t(i1(n, "triangles", 0));
  size_t at = size_t(std::stoull(n.attribute("offset").value()));
  const float *P = blob_at<float>(file, at, nv * 3);
  at += nv * 3 * sizeof(float);
  const float *N = blob_at<float>(file, at, nv * 3);
  at += nv * 3 * sizeof(float);
  const float *UV = blob_at<float>(file, at, nv * 2);
  at += nv * 2 * sizeof(float);
  const float *UV2 = nullptr; /* (the backs' texture coordinates, when some faces show both sides) */
  if (i1(n, "uv2", 0)) {
    UV2 = blob_at<float>(file, at, nv * 2);
    at += nv * 2 * sizeof(float);
  }
  const int32_t *tri = blob_at<int32_t>(file, at, nt * 3);
  at += nt * 3 * sizeof(int32_t);
  const uint16_t *slot = blob_at<uint16_t>(file, at, nt);

  Mesh *mesh = scene->create_node<Mesh>();
  mesh->name = ustring(n.attribute("name").value());

  /* Its materials: indices into the scene's, in the order its triangles name them. */
  array<Node *> used;
  const std::vector<float> mats = floats(n.attribute("materials").value());
  for (const float f : mats) {
    const int i = int(f);
    used.push_back_slow(i >= 0 && i < int(built.materials.size()) ? built.materials[size_t(i)] : scene->default_surface);
  }
  if (used.empty()) {
    used.push_back_slow(scene->default_surface);
  }
  /* (Cycles' setter takes the array's contents, leaving it empty: count them first. Before 1.0.5
   * every triangle was given the mesh's first material, so a group painted in several, a picture
   * in its frame, showed the first all over.) */
  const size_t used_count = used.size();
  mesh->set_used_shaders(used);

  mesh->resize_mesh(int(nv), int(nt));
  packed_float3 *pos = mesh->get_position_for_write();
  for (size_t v = 0; v < nv; v++) {
    pos[v] = make_float3(P[v * 3], P[v * 3 + 1], P[v * 3 + 2]);
  }
  int *triangles = mesh->get_triangles().data();
  for (size_t t = 0; t < nt * 3; t++) {
    const int32_t i = tri[t];
    triangles[t] = (i >= 0 && size_t(i) < nv) ? i : 0;
  }
  array<int> &shaders = mesh->get_shader();
  for (size_t t = 0; t < nt; t++) {
    shaders[t] = slot[t] < used_count ? int(slot[t]) : 0;
  }
  std::ranges::fill(mesh->get_smooth(), true);
  mesh->tag_triangles_modified();
  mesh->tag_shader_modified();
  mesh->tag_smooth_modified();

  Attribute *normals = mesh->attributes.add(ATTR_STD_VERTEX_NORMAL);
  packed_normal *nd = normals->data_for_write<packed_normal>();
  for (size_t v = 0; v < nv; v++) {
    nd[v] = packed_normal(make_float3(N[v * 3], N[v * 3 + 1], N[v * 3 + 2]));
  }

  /* Texture coordinates, at each corner of each triangle. */
  Attribute *uv = mesh->attributes.add(ATTR_STD_UV, ustring("UVMap"));
  float2 *ud = uv->data_for_write<float2>();
  for (size_t t = 0; t < nt * 3; t++) {
    const size_t v = size_t(triangles[t]);
    ud[t] = make_float2(UV[v * 2], UV[v * 2 + 1]);
  }
  if (UV2 != nullptr) {
    Attribute *back = mesh->attributes.add(ustring("UVBack"), TypeFloat2, ATTR_ELEMENT_CORNER);
    float2 *bd = back->data_for_write<float2>();
    for (size_t t = 0; t < nt * 3; t++) {
      const size_t v = size_t(triangles[t]);
      bd[t] = make_float2(UV2[v * 2], UV2[v * 2 + 1]);
    }
  }
  return mesh;
}

/* The lamps' and the sun's shader: plain white light, as strong as the light says. (Cycles' own
 * default_light shader is an emission of strength 0 — a light given it gives no light at all.)
 * One for each scene built: 1.0.7 (it was made once for the whole program; since 1.0.5 the
 * brightness is measured in a first scene of its own, and the render's scene then used the
 * first's shader after it was freed: about one render in three stopped with an access
 * violation, no picture and no word). */
static Shader *light_shader_made = nullptr;

static Shader *light_shader(Scene *scene)
{
  Shader *&made = light_shader_made;
  if (made == nullptr) {
    unique_ptr<ShaderGraph> graph = make_unique<ShaderGraph>();
    EmissionNode *emission = graph->create_node<EmissionNode>();
    emission->set_color(one_float3());
    emission->set_strength(1.0f);
    graph->connect(emission->output("Emission"), graph->output()->input("Surface"));
    made = scene->create_node<Shader>();
    made->name = "akar_light";
    made->set_graph(std::move(graph));
    made->tag_update(scene);
  }
  return made;
}

static void make_light(Scene *scene, const xml_node &n)
{
  const std::string type = n.attribute("type").value();
  const float3 colour = f3(n, "color", one_float3());
  const float power = f1(n, "power", 10.0f); /* watts, as Blender's */
  const float3 pos = f3(n, "position", zero_float3());
  const float3 dir = f3(n, "dir", make_float3(0.0f, 0.0f, -1.0f));
  Light *light = nullptr;
  if (type == "spot") {
    SpotLight *spot = scene->create_node<SpotLight>();
    spot->set_radius(f1(n, "radius", 0.02f));
    spot->set_angle(f1(n, "spot_angle", 0.8f));
    spot->set_smooth(f1(n, "spot_smooth", 0.3f));
    light = spot;
  }
  else if (type == "area") {
    AreaLight *area = scene->create_node<AreaLight>();
    area->set_sizeu(f1(n, "size_u", 0.3f));
    area->set_sizev(f1(n, "size_v", 0.3f));
    area->set_ellipse(i1(n, "ellipse", 0) != 0);
    area->set_spread(f1(n, "spread", float(M_PI)));
    light = area;
  }
  else {
    PointLight *point = scene->create_node<PointLight>();
    point->set_radius(f1(n, "radius", 0.02f));
    light = point;
  }
  light->set_strength(colour * power);
  light->set_cast_shadow(true);
  light->set_use_mis(true);
  array<Node *> used;
  used.push_back_slow(light_shader(scene));
  light->set_used_shaders(used);
  Object *object = scene->create_node<Object>();
  object->set_tfm(facing(pos, dir, f3(n, "axis_u", zero_float3())));
  object->set_visibility(PATH_RAY_VISIBILITY_ALL & ~PATH_RAY_VISIBILITY_CAMERA);
  object->set_geometry(light);
}

static void make_world(Scene *scene, const std::string &folder, const xml_node &world, const xml_node &sun)
{
  Shader *bg = scene->default_background;
  unique_ptr<ShaderGraph> graph = make_unique<ShaderGraph>();
  BackgroundNode *back = graph->create_node<BackgroundNode>();
  back->set_strength(f1(world, "strength", 1.0f));
  const std::string kind = world.attribute("kind").value();
  if (kind == "sky") {
    SkyTextureNode *sky = graph->create_node<SkyTextureNode>();
    sky->set_sky_type(NODE_SKY_HOSEK);
    sky->set_sun_direction(safe_normalize(f3(world, "sun_dir", make_float3(0.3f, -0.5f, 0.8f))));
    sky->set_turbidity(f1(world, "turbidity", 2.5f));
    sky->set_ground_albedo(f1(world, "ground_albedo", 0.3f));
    graph->connect(sky->output("Color"), back->input("Color"));
  }
  else if (kind == "texture") {
    /* A 360-degree picture (equirectangular .hdr or .exr), turned about the vertical by
     * "rotation" (radians, anticlockwise seen from above). */
    TextureCoordinateNode *coords = graph->create_node<TextureCoordinateNode>();
    MappingNode *turn = graph->create_node<MappingNode>();
    turn->set_mapping_type(NODE_MAPPING_TYPE_POINT);
    turn->set_rotation(make_float3(0.0f, 0.0f, -f1(world, "rotation", 0.0f)));
    EnvironmentTextureNode *env = graph->create_node<EnvironmentTextureNode>();
    env->set_filename(ustring(path_join(folder, world.attribute("texture").value())));
    env->set_colorspace(u_colorspace_data); /* (.hdr and .exr are linear already) */
    env->set_interpolation(INTERPOLATION_LINEAR);
    env->set_projection(NODE_ENVIRONMENT_EQUIRECTANGULAR);
    graph->connect(coords->output("Generated"), turn->input("Vector"));
    graph->connect(turn->output("Vector"), env->input("Vector"));
    graph->connect(env->output("Color"), back->input("Color"));
  }
  else {
    back->set_color(f3(world, "color", make_float3(0.8f, 0.85f, 0.9f)));
  }
  ShaderOutput *surface = back->output("Background");
  if (!world.attribute("camera_color").empty()) {
    /* What the camera sees straight on is a plain colour; the light still comes from the sky. */
    BackgroundNode *plain = graph->create_node<BackgroundNode>();
    plain->set_color(f3(world, "camera_color", one_float3()));
    plain->set_strength(f1(world, "camera_strength", 1.0f));
    LightPathNode *path = graph->create_node<LightPathNode>();
    MixClosureNode *mix = graph->create_node<MixClosureNode>();
    graph->connect(path->output("Is Camera Ray"), mix->input("Fac"));
    graph->connect(back->output("Background"), mix->input("Closure1"));
    graph->connect(plain->output("Background"), mix->input("Closure2"));
    surface = mix->output("Closure");
  }
  graph->connect(surface, graph->output()->input("Surface"));
  bg->set_graph(std::move(graph));
  bg->tag_update(scene);

  /* The sky as a light as well. Cycles looks for the background directly (and by how bright each
   * part of it is) only through a background light; without one, a room lit through its windows
   * is found by the sky only when a bounce happens to go out of a window: grainy, and the
   * denoiser turns the grain into smudges and scratches. (A plain colour needs none: Cycles
   * leaves it off.) */
  {
    BackgroundLight *sky = scene->create_node<BackgroundLight>();
    sky->set_map_resolution(i1(world, "map_resolution", 0)); /* (0: as Cycles sees fit) */
    sky->set_use_mis(true);
    array<Node *> used;
    used.push_back_slow(bg);
    sky->set_used_shaders(used);
    Object *object = scene->create_node<Object>();
    object->set_tfm(transform_identity());
    object->set_visibility(PATH_RAY_VISIBILITY_ALL & ~PATH_RAY_VISIBILITY_CAMERA);
    object->set_geometry(sky);
  }

  if (sun) {
    SunLight *light = scene->create_node<SunLight>();
    light->set_strength(f3(sun, "color", one_float3()) * f1(sun, "strength", 4.0f));
    light->set_angle(f1(sun, "angle", 0.0105f));
    light->set_cast_shadow(true);
    light->set_use_mis(true);
    array<Node *> used;
    used.push_back_slow(light_shader(scene));
    light->set_used_shaders(used);
    Object *object = scene->create_node<Object>();
    /* "dir" points to the sun; the light travels the other way. */
    object->set_tfm(facing(zero_float3(), -f3(sun, "dir", make_float3(0.3f, -0.5f, 0.8f))));
    object->set_visibility(PATH_RAY_VISIBILITY_ALL & ~PATH_RAY_VISIBILITY_CAMERA);
    object->set_geometry(light);
  }
}

static void make_camera(Scene *scene, const xml_node &c, const int width, const int height)
{
  Camera *cam = scene->camera;
  cam->set_full_width(width);
  cam->set_full_height(height);
  const float3 eye = f3(c, "eye", zero_float3());
  const float3 right = safe_normalize(f3(c, "right", make_float3(1, 0, 0)));
  const float3 up = safe_normalize(f3(c, "up", make_float3(0, 0, 1)));
  const float3 fwd = safe_normalize(f3(c, "forward", make_float3(0, 1, 0)));
  /* Cycles' camera looks along its +Z, with +X right and +Y up. */
  Transform t;
  t.x = make_float4(right.x, up.x, fwd.x, eye.x);
  t.y = make_float4(right.y, up.y, fwd.y, eye.y);
  t.z = make_float4(right.z, up.z, fwd.z, eye.z);
  cam->set_matrix(t);
  cam->set_nearclip(f1(c, "clip_start", 0.01f));
  cam->set_farclip(f1(c, "clip_end", 10000.0f));
  const float aspect = float(width) / float(height);
  if (i1(c, "ortho", 0)) {
    cam->set_camera_type(CAMERA_ORTHOGRAPHIC);
    const float h = f1(c, "ortho_half_height", 5.0f);
    cam->set_viewplane_left(-h * aspect);
    cam->set_viewplane_right(h * aspect);
    cam->set_viewplane_bottom(-h);
    cam->set_viewplane_top(h);
  }
  else {
    cam->set_camera_type(CAMERA_PERSPECTIVE);
    /* Akar's field of view is the picture's height; Cycles' is its shorter side. */
    const float fov_y = f1(c, "fov_y", 0.61f);
    const float shift = f1(c, "shift_y", 0.0f);
    float vx = aspect, vy = 1.0f, fov = fov_y;
    if (aspect < 1.0f) {
      /* Portrait: Cycles' fov is across the width. */
      fov = 2.0f * atanf(tanf(fov_y * 0.5f) * aspect);
      vx = 1.0f;
      vy = 1.0f / aspect;
    }
    cam->set_fov(fov);
    cam->set_viewplane_left(-vx);
    cam->set_viewplane_right(vx);
    cam->set_viewplane_bottom(-vy - shift * vy);
    cam->set_viewplane_top(vy - shift * vy);
  }
  cam->need_flags_update = true;
  cam->update(scene);
}

static Built build_scene(Scene *scene, const SceneFile &file)
{
  Built built;
  light_shader_made = nullptr; /* (this scene's own, made when the first light needs it) */
  const xml_node root = file.doc.child("akar-render");
  if (!root) {
    fail("the scene description has no <akar-render>");
  }
  const xml_node film = root.child("film");
  built.width = std::max(16, i1(film, "width", 1280));
  built.height = std::max(16, i1(film, "height", 720));
  scene->film->set_exposure(1.0f);
  scene->film->set_filter_width(f1(film, "filter_width", 1.5f));

  std::vector<std::pair<std::string, std::string>> textures;
  for (xml_node t = root.child("texture"); t; t = t.next_sibling("texture")) {
    textures.emplace_back(t.attribute("path").value(), t.attribute("colorspace").value());
  }
  std::vector<xml_node> material_nodes;
  for (xml_node m = root.child("material"); m; m = m.next_sibling("material")) {
    material_nodes.push_back(m);
  }
  for (const xml_node &m : material_nodes) {
    built.materials.push_back(make_material(scene, file, m, textures, material_nodes));
  }
  for (xml_node n = root.child("mesh"); n; n = n.next_sibling("mesh")) {
    built.meshes.push_back(make_mesh(scene, file, n, built));
  }
  size_t objects = 0;
  for (xml_node o = root.child("object"); o; o = o.next_sibling("object")) {
    const int m = i1(o, "mesh", -1);
    if (m < 0 || m >= int(built.meshes.size())) {
      continue;
    }
    Object *object = scene->create_node<Object>();
    object->set_geometry(built.meshes[size_t(m)]);
    object->set_tfm(tfm12(o, "tfm"));
    objects++;
  }
  size_t lights = 0;
  for (xml_node l = root.child("light"); l; l = l.next_sibling("light")) {
    make_light(scene, l);
    lights++;
  }
  make_world(scene, file.folder, root.child("world"), root.child("sun"));
  make_camera(scene, root.child("camera"), built.width, built.height);
  say(string_printf("SCENE %zu materials %zu meshes %zu objects %zu lights",
                    built.materials.size(), built.meshes.size(), objects, lights));
  return built;
}

/* ---------------------------------------------------------------------------------------- */
/* Writing the picture                                                                       */
/* ---------------------------------------------------------------------------------------- */

class PictureDriver : public OutputDriver {
 public:
  PictureDriver(std::string folder, std::atomic<int> *samples) : folder_(std::move(folder)), samples_(samples) {}

  void write_render_tile(const Tile &tile) override
  {
    if (measuring) {
      measure(tile);
      return;
    }
    if (write(tile, "final.akri")) {
      final_written = true;
    }
  }

  bool update_render_tile(const Tile &tile) override
  {
    if (measuring) {
      return false;
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - last_ < std::chrono::milliseconds(900)) {
      return false;
    }
    last_ = now;
    if (write(tile, "preview.akri")) {
      say("PREVIEW " + path_join(folder_, "preview.akri"));
      return true;
    }
    return false;
  }

  bool final_written = false;

  /* 1.0.5: the first, small picture only measures how bright the model is (see main). */
  std::atomic<bool> measuring{false};
  /* The model's middle brightness in it (-1: none seen). */
  float measured = -1.0f;
  /* 1.0.7: the colour of the light falling on the model in it (its diffuse light, without the
   * surfaces' own colours), its brightness 1; (1, 1, 1) when none was seen. */
  float3 white = one_float3();
  /* The film's exposure in the render, taken off again as the picture is written (the file
   * holds the light as it is: Akar exposes it). */
  float exposure = 1.0f;

 private:
  void measure(const Tile &tile)
  {
    const int w = tile.size.x, h = tile.size.y;
    std::vector<float> px(size_t(w) * size_t(h) * 4);
    std::vector<float> depth(size_t(w) * size_t(h));
    if (!tile.get_pass_pixels("combined", 4, px.data())) {
      return;
    }
    const bool has_depth = tile.get_pass_pixels("depth", 1, depth.data());
    std::vector<float> model, all;
    for (size_t i = 0; i < size_t(w) * size_t(h); i++) {
      const float y = 0.2126f * px[i * 4] + 0.7152f * px[i * 4 + 1] + 0.0722f * px[i * 4 + 2];
      if (!std::isfinite(y) || y < 0.0f) {
        continue; /* (a pixel no light reached counts: leaving them out made the room look brighter than it is) */
      }
      all.push_back(y);
      if (has_depth && depth[i] > 0.0f && depth[i] < 1e8f) {
        model.push_back(y);
      }
    }
    std::vector<float> &use = model.size() > 16 ? model : all;
    if (use.empty()) {
      return;
    }
    double sum = 0.0;
    for (const float y : use) {
      sum += y;
    }
    std::nth_element(use.begin(), use.begin() + use.size() / 2, use.end());
    const float middle = use[use.size() / 2];
    measured = middle > 0.0f ? middle : float(sum / double(use.size()));

    /* The light's colour: all the diffuse light reaching the model (direct and bounced), added
     * up, so the brightest-lit surfaces count most, as they do to the eye. */
    std::vector<float> direct(size_t(w) * size_t(h) * 3), indirect(size_t(w) * size_t(h) * 3);
    if (tile.get_pass_pixels("diffuse_direct", 3, direct.data()) && tile.get_pass_pixels("diffuse_indirect", 3, indirect.data())) {
      double r = 0.0, g = 0.0, b = 0.0;
      for (size_t i = 0; i < size_t(w) * size_t(h); i++) {
        if (has_depth && !(depth[i] > 0.0f && depth[i] < 1e8f)) {
          continue;
        }
        const float lr = direct[i * 3] + indirect[i * 3];
        const float lg = direct[i * 3 + 1] + indirect[i * 3 + 1];
        const float lb = direct[i * 3 + 2] + indirect[i * 3 + 2];
        if (!std::isfinite(lr) || !std::isfinite(lg) || !std::isfinite(lb) || lr < 0.0f || lg < 0.0f || lb < 0.0f || lr + lg + lb > 1e6f) {
          continue;
        }
        r += lr;
        g += lg;
        b += lb;
      }
      const double y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      if (y > 1e-12) {
        white = make_float3(float(r / y), float(g / y), float(b / y));
      }
    }
  }

  bool write(const Tile &tile, const char *name)
  {
    if (!(tile.size == tile.full_size)) {
      return false; /* (whole pictures only) */
    }
    const int w = tile.size.x, h = tile.size.y;
    std::vector<float> px(size_t(w) * size_t(h) * 4);
    if (!tile.get_pass_pixels("combined", 4, px.data())) {
      return false;
    }
    if (exposure != 1.0f) {
      const float k = 1.0f / exposure;
      for (size_t i = 0; i < size_t(w) * size_t(h); i++) {
        px[i * 4] *= k;
        px[i * 4 + 1] *= k;
        px[i * 4 + 2] *= k;
      }
    }
    /* The alpha says where the model is (1) and where the sky or background is seen (0), from
     * the depth: Akar's automatic exposure looks at the model, not at what's outside. */
    std::vector<float> depth(size_t(w) * size_t(h));
    if (tile.get_pass_pixels("depth", 1, depth.data())) {
      for (size_t i = 0; i < depth.size(); i++) {
        px[i * 4 + 3] = (depth[i] > 0.0f && depth[i] < 1e8f) ? 1.0f : 0.0f;
      }
    }
    const std::string path = path_join(folder_, name);
    const std::string temp = path + ".part";
    {
      std::ofstream out(temp, std::ios::binary | std::ios::trunc);
      if (!out) {
        return false;
      }
      const int32_t header[4] = {int32_t(w), int32_t(h), int32_t(samples_ ? samples_->load() : 0), 0};
      out.write("AKRI", 4);
      out.write(reinterpret_cast<const char *>(header), sizeof(header));
      /* Cycles' rows run from the bottom; the file's from the top. */
      for (int y = h - 1; y >= 0; y--) {
        out.write(reinterpret_cast<const char *>(px.data() + size_t(y) * size_t(w) * 4), std::streamsize(size_t(w) * 4 * sizeof(float)));
      }
      if (!out) {
        return false;
      }
    }
    std::remove(path.c_str());
    return std::rename(temp.c_str(), path.c_str()) == 0;
  }

  std::string folder_;
  std::atomic<int> *samples_;
  std::chrono::steady_clock::time_point last_{};
};

/* ---------------------------------------------------------------------------------------- */

static DeviceInfo choose_device(const std::string &wanted)
{
  const char *order[] = {"OPTIX", "CUDA", "CPU"};
  for (const char *name : order) {
    if (wanted != "AUTO" && wanted != name) {
      continue;
    }
    const DeviceType type = Device::type_from_string(name);
    const vector<DeviceInfo> devices = Device::available_devices(DEVICE_MASK(type));
    if (!devices.empty()) {
      return devices.front();
    }
  }
  const vector<DeviceInfo> cpu = Device::available_devices(DEVICE_MASK_CPU);
  if (cpu.empty()) {
    fail("no device to render on");
  }
  return cpu.front();
}

CCL_NAMESPACE_END

using namespace ccl;

int main(const int argc, const char **argv)
{
  log_init(nullptr);
  path_init();
  system_max_open_files_ensure();

  std::string scene_path, out_folder, device_name = "AUTO";
  int samples = 512, threads = 0;
  float threshold = 0.01f;
  double time_limit = 0.0;
  /* 1.0.5: glossy light after a bounce softened, as Blender has it by default (its "Filter
   * Glossy" of 1): window light bouncing off a shiny floor or cabinet into the shade is otherwise
   * found by chance, and is grain there. */
  float filter_glossy = 1.0f;
  bool caustics = true;
  /* 1.0.5: Cycles judges a pixel clean enough by its light before any exposure, as if it were
   * shown as it is; a room lit through its windows can be a hundred times darker than its picture
   * shows it, so its shade was taken for clean after a few dozen samples, grainy, and the noise
   * remover smoothed the grain into scratches. A first small picture (an eighth across, 16
   * samples, in a session of its own) measures how bright the model is, and the render is
   * exposed to that (the exposure is taken off again in the file). --exposure e sets it instead
   * (0: measured). */
  float film_exposure = 0.0f;
  bool denoise = true, list = false;
  for (int i = 1; i < argc; i++) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
    if (a == "--device") {
      device_name = next();
      for (char &ch : device_name) {
        ch = char(std::toupper(static_cast<unsigned char>(ch)));
      }
    }
    else if (a == "--samples") samples = std::max(1, std::atoi(next().c_str()));
    else if (a == "--threshold") threshold = float(std::atof(next().c_str()));
    else if (a == "--time-limit") time_limit = std::atof(next().c_str());
    else if (a == "--threads") threads = std::atoi(next().c_str());
    else if (a == "--no-denoise") denoise = false;
    else if (a == "--filter-glossy") filter_glossy = std::max(0.0f, float(std::atof(next().c_str())));
    else if (a == "--no-caustics") caustics = false;
    else if (a == "--exposure") film_exposure = std::max(0.0f, float(std::atof(next().c_str())));
    else if (a == "--out") out_folder = next();
    else if (a == "--list-devices") list = true;
    else if (a == "--version") { say(std::string("akar-render 1 (Cycles ") + CYCLES_VERSION_STRING + ")"); return 0; }
    else scene_path = a;
  }
  if (list) {
    for (const DeviceInfo &info : Device::available_devices()) {
      say("DEVICE " + Device::string_from_type(info.type) + " " + info.description);
    }
    return 0;
  }
  if (scene_path.empty()) {
    say("usage: akar-render <scene.akrs> [--out folder] [--device auto|optix|cuda|cpu] [--samples n] [--threshold t] [--time-limit s] [--no-denoise] [--filter-glossy f] [--no-caustics] [--exposure e]");
    return 2;
  }
  if (out_folder.empty()) {
    out_folder = path_dirname(scene_path);
  }

  SceneFile file;
  read_scene_file(scene_path, file);

  SessionParams sp;
  sp.device = choose_device(device_name);
  sp.background = true;
  sp.samples = samples;
  sp.threads = threads;
  sp.time_limit = time_limit;
  sp.use_auto_tile = false;
  sp.tile_size = 0;
  SceneParams scp;
  scp.shadingsystem = SHADINGSYSTEM_SVM;
  say("DEVICE " + Device::string_from_type(sp.device.type) + " " + sp.device.description);

  /* A session with the scene built in it. 1.0.5: twice when the brightness is measured, a
   * session each (one session rendering twice, with the exposure changed between, left the
   * second picture without its noise removed). */
  std::atomic<int> done_samples{0};
  std::mutex session_mutex; /* (the "stop" listener and the change of session) */
  unique_ptr<Session> session;
  PictureDriver *picture = nullptr;
  Built built;
  const auto started = std::chrono::steady_clock::now();
  auto seconds = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count(); };
  auto open_session = [&](const float exposure, const bool measuring) {
    auto made = make_unique<Session>(sp, scp);
    auto driver = make_unique<PictureDriver>(out_folder, &done_samples);
    PictureDriver *made_picture = driver.get();
    made_picture->measuring = measuring;
    made_picture->exposure = exposure;
    made->set_output_driver(std::move(driver));
    Scene *scene = made->scene.get();
    {
      const thread_scoped_lock lock(scene->mutex);
      built = build_scene(scene, file);
      Integrator *in = scene->integrator;
      in->set_use_adaptive_sampling(true);
      in->set_adaptive_threshold(threshold);
      in->set_max_bounce(12);
      in->set_max_diffuse_bounce(4);
      in->set_max_glossy_bounce(4);
      in->set_max_transmission_bounce(8);
      in->set_transparent_max_bounce(16);
      in->set_sample_clamp_indirect(10.0f);
      in->set_filter_glossy(filter_glossy);
      in->set_caustics_reflective(caustics);
      in->set_caustics_refractive(caustics);
      in->set_use_light_tree(true);
      in->set_use_denoise(denoise || measuring);
      in->set_denoiser_type(DENOISER_OPENIMAGEDENOISE);
      in->set_denoiser_prefilter(DENOISER_PREFILTER_ACCURATE);
      scene->film->set_exposure(exposure);
      Pass *pass = scene->create_node<Pass>();
      pass->set_name(ustring("combined"));
      pass->set_type(PASS_COMBINED);
      Pass *depth = scene->create_node<Pass>();
      depth->set_name(ustring("depth"));
      depth->set_type(PASS_DEPTH);
      if (measuring) {
        /* 1.0.7: the light alone (Cycles divides out the surfaces' colours), for its colour. */
        for (const auto &[name, type] : {std::pair{"diffuse_direct", PASS_DIFFUSE_DIRECT}, std::pair{"diffuse_indirect", PASS_DIFFUSE_INDIRECT}}) {
          Pass *light = scene->create_node<Pass>();
          light->set_name(ustring(name));
          light->set_type(type);
        }
      }
    }
    Session *raw = made.get();
    made->progress.set_update_callback([&, raw, made_picture]() {
      static std::chrono::steady_clock::time_point last{};
      const int s = raw->progress.get_current_sample();
      if (made_picture->measuring) {
        return;
      }
      done_samples = s; /* (every time: the final picture is written with it) */
      const auto now = std::chrono::steady_clock::now();
      if (now - last < std::chrono::milliseconds(250)) {
        return;
      }
      last = now;
      double total = 0.0, render = 0.0;
      raw->progress.get_time(total, render);
      const double p = raw->progress.get_progress();
      const double left = p > 0.02 ? total * (1.0 - p) / p : -1.0;
      say(string_printf("PROGRESS %.4f %d %d %.1f %.1f", p, s, samples, seconds(), left));
    });
    const std::lock_guard<std::mutex> guard(session_mutex);
    session = std::move(made);
    picture = made_picture;
  };
  auto close_session = [&]() {
    const std::lock_guard<std::mutex> guard(session_mutex);
    session.reset();
    picture = nullptr;
  };

  /* "stop" on standard input ends the render, keeping what's done. */
  std::atomic<bool> stop_asked{false};
  std::thread listener([&]() {
    std::string line;
    while (std::getline(std::cin, line)) {
      if (line.rfind("stop", 0) == 0) {
        stop_asked = true;
        const std::lock_guard<std::mutex> guard(session_mutex);
        if (session && picture && !picture->measuring) {
          session->set_time_limit(0.001);
          session->set_samples(std::max(1, session->progress.get_current_sample()));
        }
        break;
      }
    }
  });
  listener.detach();

  /* How bright the model is: a small picture first (an eighth across, 16 samples, its noise
   * removed), unless the exposure was given. */
  float exposure = film_exposure;
  if (exposure <= 0.0f) {
    open_session(1.0f, true);
    SessionParams measure_params = sp;
    measure_params.samples = 16;
    measure_params.time_limit = 0.0;
    BufferParams mb;
    mb.width = std::max(16, built.width / 8);
    mb.height = std::max(16, built.height / 8);
    mb.full_width = mb.width;
    mb.full_height = mb.height;
    session->reset(measure_params, mb);
    session->start();
    session->wait();
    const std::string measure_error = session->progress.get_error_message();
    const float measured = picture->measured;
    const float3 white = picture->white;
    close_session();
    if (!measure_error.empty()) {
      fail(measure_error);
    }
    exposure = measured > 0.0f ? std::clamp(0.18f / measured, 0.001f, 100000.0f) : 1.0f;
    say(string_printf("EXPOSURE %.6g %.6g %.1f", exposure, measured, seconds()));
    say(string_printf("WHITE %.5f %.5f %.5f", white.x, white.y, white.z));
  }
  if (stop_asked) {
    sp.samples = 1; /* (stopped while measuring: a picture at once) */
  }
  open_session(exposure, false);

  BufferParams bp;
  bp.width = built.width;
  bp.height = built.height;
  bp.full_width = built.width;
  bp.full_height = built.height;
  session->reset(sp, bp);
  session->start();
  session->wait();

  const std::string error = session->progress.get_error_message();
  const bool written = picture->final_written;
  session.reset();
  if (!error.empty()) {
    fail(error);
  }
  if (!written) {
    fail("the render ended without a picture");
  }
  say(string_printf("DONE %s %.1f", path_join(out_folder, "final.akri").c_str(), seconds()));
  quit(0);
}
