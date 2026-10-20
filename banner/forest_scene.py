import json, math, sys
from pathlib import Path

import bpy
from mathutils import Vector

cfg = json.loads(Path(sys.argv[-1]).read_text())
A = Path(cfg["assets"])
OUT = Path(cfg["out"])
LIB = A / "fir_trees.blend"


def build_library():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(A / "fir_tree_01" / "fir_tree_01_1k.gltf"))
    for o in [o for o in bpy.data.objects if o.name.startswith("fir_tree_01_a")]:
        me = o.data
        bpy.data.objects.remove(o)
        bpy.data.meshes.remove(me)
    alpha = bpy.data.images.load(str(A / "fir_tree_01" / "textures" / "alpha.png"))
    alpha.colorspace_settings.name = "Non-Color"
    alpha.pack()
    for m in bpy.data.materials:
        nt = m.node_tree
        b = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if b is None:
            continue
        for l in list(b.inputs["Metallic"].links):
            nt.links.remove(l)
        b.inputs["Metallic"].default_value = 0
        b.inputs["Specular IOR Level"].default_value = .25
        if "twig" not in m.name.lower() or not b.inputs["Base Color"].links:
            continue
        b.inputs["Specular IOR Level"].default_value = .04
        for l in list(b.inputs["Roughness"].links):
            nt.links.remove(l)
        b.inputs["Roughness"].default_value = .85
        col = b.inputs["Base Color"].links[0].from_node
        green = nt.nodes.new("ShaderNodeMix")
        green.data_type = "RGBA"
        green.blend_type = "MULTIPLY"
        green.inputs[0].default_value = 1
        green.inputs[7].default_value = (.42, .72, .4, 1)
        nt.links.new(col.outputs[0], green.inputs[6])
        nt.links.new(green.outputs[2], b.inputs["Base Color"])
        at = nt.nodes.new("ShaderNodeTexImage")
        at.image = alpha
        if col.inputs[0].links:
            nt.links.new(col.inputs[0].links[0].from_socket, at.inputs[0])
        out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
        tr = nt.nodes.new("ShaderNodeBsdfTranslucent")
        nt.links.new(green.outputs[2], tr.inputs["Color"])
        mix = nt.nodes.new("ShaderNodeMixShader")
        mix.inputs[0].default_value = .25
        nt.links.new(b.outputs[0], mix.inputs[1])
        nt.links.new(tr.outputs[0], mix.inputs[2])
        cut = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(at.outputs["Color"], cut.inputs[0])
        nt.links.new(nt.nodes.new("ShaderNodeBsdfTransparent").outputs[0], cut.inputs[1])
        nt.links.new(mix.outputs[0], cut.inputs[2])
        for l in list(b.inputs["Alpha"].links):
            nt.links.remove(l)
        nt.links.new(cut.outputs[0], out.inputs["Surface"])
    for key in ("fir_tree_01_c", "fir_tree_01_b"):
        col = bpy.data.collections.new(key)
        bpy.context.scene.collection.children.link(col)
        for o in [o for o in bpy.data.objects if o.name.startswith(key)]:
            for c in o.users_collection:
                c.objects.unlink(o)
            o.location = (0, 0, 0)
            col.objects.link(o)
    for im in bpy.data.images:
        if im.filepath and not im.packed_file:
            im.pack()
    bpy.ops.wm.save_as_mainfile(filepath=str(LIB), compress=True)


if not LIB.exists():
    build_library()

bpy.ops.wm.read_factory_settings(use_empty=True)
with bpy.data.libraries.load(str(LIB), link=False) as (src, dst):
    dst.collections = [c for c in src.collections if c.startswith("fir_tree_01")]
trees = list(dst.collections)

sc = bpy.context.scene
sc.render.engine = "CYCLES"
cy = sc.cycles
cy.device = "CPU"
cy.samples = cfg["samples"]
cy.use_adaptive_sampling = True
cy.adaptive_threshold = .03
cy.use_denoising = True
cy.denoiser = "OPENIMAGEDENOISE"
cy.max_bounces = 4
cy.diffuse_bounces = 2
cy.glossy_bounces = 1
cy.transparent_max_bounces = 32
cy.sample_clamp_indirect = 3
sc.render.resolution_x, sc.render.resolution_y = cfg["res"]
sc.render.resolution_percentage = 100
sc.render.film_transparent = True
sc.view_settings.view_transform = "Standard"
sc.view_settings.look = "None"
sc.view_settings.exposure = cfg["exposure"]
sc.render.image_settings.file_format = "PNG"
sc.render.image_settings.color_mode = "RGBA"
sc.render.image_settings.color_depth = "16"


def grid_mesh(name, z, x0, y0, dx, dy):
    ny, nx = len(z), len(z[0])
    me = bpy.data.meshes.new(name)
    verts = [(x0 + i * dx, y0 + j * dy, z[j][i]) for j in range(ny) for i in range(nx)]
    faces = [(j * nx + i, j * nx + i + 1, (j + 1) * nx + i + 1, (j + 1) * nx + i)
             for j in range(ny - 1) for i in range(nx - 1)]
    me.from_pydata(verts, [], faces)
    me.update()
    for p in me.polygons:
        p.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    sc.collection.objects.link(ob)
    return ob


g = cfg["ground"]
ground = grid_mesh("ground", g["z"], g["x0"], g["y0"], g["dx"], g["dy"])
mat = bpy.data.materials.new("ground")
nt = mat.node_tree
b = nt.nodes["Principled BSDF"]
b.inputs["Roughness"].default_value = .95
b.inputs["Specular IOR Level"].default_value = .1
tc = nt.nodes.new("ShaderNodeTexCoord")
mp = nt.nodes.new("ShaderNodeMapping")
mp.inputs["Scale"].default_value = (1 / cfg["ground_tex_scale"],) * 3
nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
tex = nt.nodes.new("ShaderNodeTexImage")
tex.image = bpy.data.images.load(str(A / "tex" / cfg["ground_tex"] / "diff.jpg"))
tex.projection = "BOX"
nt.links.new(mp.outputs[0], tex.inputs[0])
tint = nt.nodes.new("ShaderNodeMix")
tint.data_type = "RGBA"
tint.blend_type = "MULTIPLY"
tint.inputs[0].default_value = 1
tint.inputs[7].default_value = (*cfg["ground_tint"], 1)
nt.links.new(tex.outputs[0], tint.inputs[6])
nt.links.new(tint.outputs[2], b.inputs["Base Color"])
ground.data.materials.append(mat)


def scatter(name, pts):
    me = bpy.data.meshes.new(name)
    me.from_pydata([p[:3] for p in pts], [], [])
    sa = me.attributes.new("s", "FLOAT", "POINT")
    ra = me.attributes.new("r", "FLOAT", "POINT")
    ia = me.attributes.new("idx", "INT", "POINT")
    for k, p in enumerate(pts):
        sa.data[k].value, ra.data[k].value, ia.data[k].value = p[3], p[4], int(p[5])
    ob = bpy.data.objects.new(name, me)
    sc.collection.objects.link(ob)
    parent = bpy.data.collections.new(name + "_set")
    for c in trees:
        parent.children.link(c)
    ng = bpy.data.node_groups.new(name, "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    N, Lk = ng.nodes, ng.links
    gi, go = N.new("NodeGroupInput"), N.new("NodeGroupOutput")
    m2p = N.new("GeometryNodeMeshToPoints")
    ci = N.new("GeometryNodeCollectionInfo")
    ci.inputs["Collection"].default_value = parent
    ci.inputs["Separate Children"].default_value = True
    ci.inputs["Reset Children"].default_value = True
    iop = N.new("GeometryNodeInstanceOnPoints")
    iop.inputs["Pick Instance"].default_value = True

    def named(n, kind="FLOAT"):
        x = N.new("GeometryNodeInputNamedAttribute")
        x.data_type = kind
        x.inputs["Name"].default_value = n
        return x.outputs["Attribute"]

    rot = N.new("ShaderNodeCombineXYZ")
    Lk.new(named("r"), rot.inputs["Z"])
    Lk.new(gi.outputs[0], m2p.inputs["Mesh"])
    Lk.new(m2p.outputs[0], iop.inputs["Points"])
    Lk.new(ci.outputs[0], iop.inputs["Instance"])
    Lk.new(named("idx", "INT"), iop.inputs["Instance Index"])
    Lk.new(rot.outputs[0], iop.inputs["Rotation"])
    Lk.new(named("s"), iop.inputs["Scale"])
    Lk.new(iop.outputs[0], go.inputs[0])
    ob.modifiers.new("scatter", "NODES").node_group = ng
    return ob


back = scatter("back", cfg["back"])
front = scatter("front", cfg["front"])


def direction(alt, rel_az):
    e, a = math.radians(alt), math.radians(rel_az)
    return Vector((math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e)))


world = bpy.data.worlds.new("world")
sc.world = world
wn = world.node_tree
geo = wn.nodes.new("ShaderNodeTexCoord")
sep = wn.nodes.new("ShaderNodeSeparateXYZ")
wn.links.new(geo.outputs["Generated"], sep.inputs[0])
ramp = wn.nodes.new("ShaderNodeValToRGB")
ramp.color_ramp.elements[0].position = .0
ramp.color_ramp.elements[0].color = (*cfg["ground_bounce"], 1)
ramp.color_ramp.elements[1].position = .5
ramp.color_ramp.elements[1].color = (*cfg["horizon"], 1)
e = ramp.color_ramp.elements.new(1.0)
e.color = (*cfg["zenith"], 1)
mz = wn.nodes.new("ShaderNodeMapRange")
mz.inputs[1].default_value, mz.inputs[2].default_value = -1, 1
wn.links.new(sep.outputs["Z"], mz.inputs[0])
wn.links.new(mz.outputs[0], ramp.inputs[0])
bg = wn.nodes["Background"]
bg.inputs["Strength"].default_value = cfg["sky_strength"]
wn.links.new(ramp.outputs[0], bg.inputs[0])

for key in ("sun", "moon"):
    if cfg[key + "_strength"] <= 0:
        continue
    ob = bpy.data.objects.new(key, bpy.data.lights.new(key, "SUN"))
    sc.collection.objects.link(ob)
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = direction(cfg[key + "_alt"], cfg[key + "_rel_az"]).to_track_quat("Z", "Y")
    ob.data.energy = cfg[key + "_strength"]
    ob.data.color = cfg[key + "_rgb"]
    ob.data.angle = math.radians(cfg[key + "_angle"])

cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
sc.collection.objects.link(cam)
sc.camera = cam
cam.location = (0, 0, cfg["cam_z"])
cam.rotation_euler = (math.radians(90), 0, 0)
cam.data.sensor_fit = "HORIZONTAL"
cam.data.angle = math.radians(cfg["hfov"])
cam.data.shift_y = cfg["shift_y"]
cam.data.clip_end = 10_000

front.hide_render = True
sc.render.filepath = str(OUT / "back.png")
bpy.ops.render.render(write_still=True)
front.hide_render = False
back.hide_render = True
ground.hide_render = True
sc.render.filepath = str(OUT / "front.png")
bpy.ops.render.render(write_still=True)
