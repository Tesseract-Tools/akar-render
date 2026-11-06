# The scene file (.akrs), version 1

Little-endian. `"AKRS"`, `uint32 version = 1`, `uint32 xmlLength`, the XML (UTF-8), zero bytes up
to a multiple of 16 from the file's start, then the binary data. Units are metres, Z up.

```xml
<akar-render version="1">
  <film width="1920" height="1064" filter_width="1.5"/>
  <camera eye="x y z" right="x y z" up="x y z" forward="x y z" fov_y="0.785" shift_y="0"
          clip_start="0.01" clip_end="10000" ortho="0" ortho_half_height="5"/>
  <world kind="sky|color" color="r g b" strength="1" sun_dir="x y z" turbidity="2.5"/>
  <sun dir="x y z (towards the sun)" strength="4" angle="0.0105" color="r g b"/>
  <texture path="textures/0.jpg" colorspace="srgb|data"/>
  <material name="…" color="r g b (linear)" alpha="1" roughness="0.6" metallic="0" ior="1.5"
            specular="0.5" transmission="0" texture="-1" cutout="0" tint="0" tint_color="r g b"
            roughness_map="-1" metallic_from_map="0" normal_map="-1" normal_strength="1"
            emission="r g b" emission_strength="0"/>
  <mesh name="…" vertices="V" triangles="T" offset="byte offset into the data" materials="m0 m1 …"/>
  <object mesh="0" tfm="12 numbers: three rows of a 3 x 4 matrix"/>
  <light type="point|spot|area" position="x y z" dir="x y z" color="r g b" power="watts"
         radius="0.02" spot_angle="0.8" spot_smooth="0.3" size_u="0.3" size_v="0.3" ellipse="0"/>
</akar-render>
```

A mesh's data, in order: positions (V × 3 float32), normals (V × 3 float32), texture coordinates
(V × 2 float32), triangles (T × 3 int32), and each triangle's material (T × uint16, an index into
the mesh's `materials`).

The pictures akar-render writes (`preview.akri`, `final.akri`): `"AKRI"`, `int32 width, height,
samples, 0`, then width × height RGBA float32, linear, top row first.
