"""Model and render the robot, cat and owl with Blender (pip install bpy).

Each character is built from smooth primitives, lit like a studio product shot and
rendered with Cycles to a transparent PNG in figures/assets/.
"""

import argparse
import math
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parents[1]


def hex_rgba(value):
    value = value.lstrip("#")
    rgb = [int(value[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    return [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb] + [1.0]


def material(name, color, roughness=0.5, coat=0.0, subsurface=0.0, emission=None, strength=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = hex_rgba(color)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Coat Weight"].default_value = coat
    bsdf.inputs["Coat Roughness"].default_value = 0.08
    if subsurface:
        bsdf.inputs["Subsurface Weight"].default_value = subsurface
        bsdf.inputs["Subsurface Radius"].default_value = (0.4, 0.25, 0.15)
        bsdf.inputs["Subsurface Scale"].default_value = 0.08
    if emission:
        bsdf.inputs["Emission Color"].default_value = hex_rgba(emission)
        bsdf.inputs["Emission Strength"].default_value = strength
    return mat


def finish(obj, mat, smooth=True):
    obj.data.materials.append(mat)
    if smooth:
        for poly in obj.data.polygons:
            poly.use_smooth = True
    return obj


def sphere(location, scale, mat, rotation=(0, 0, 0)):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=96, ring_count=48, location=location)
    obj = bpy.context.object
    obj.scale = scale
    obj.rotation_euler = rotation
    return finish(obj, mat)


def rounded_box(location, size, radius, mat, rotation=(0, 0, 0)):
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.scale = [s / 2 for s in size]
    obj.rotation_euler = rotation
    bpy.ops.object.transform_apply(scale=True)
    bevel = obj.modifiers.new("bevel", "BEVEL")
    bevel.width = radius
    bevel.segments = 18
    bevel.limit_method = "NONE"
    return finish(obj, mat)


def cylinder(location, radius, depth, mat, rotation=(0, 0, 0), bevel=0.0):
    bpy.ops.mesh.primitive_cylinder_add(vertices=96, radius=radius, depth=depth, location=location)
    obj = bpy.context.object
    obj.rotation_euler = rotation
    if bevel:
        mod = obj.modifiers.new("bevel", "BEVEL")
        mod.width = bevel
        mod.segments = 10
        mod.limit_method = "ANGLE"
    return finish(obj, mat)


def cone(location, radius, depth, mat, rotation=(0, 0, 0), tip=0.0, scale=(1, 1, 1)):
    bpy.ops.mesh.primitive_cone_add(
        vertices=96, radius1=radius, radius2=tip, depth=depth, location=location
    )
    obj = bpy.context.object
    obj.rotation_euler = rotation
    obj.scale = scale
    sub = obj.modifiers.new("smooth", "SUBSURF")
    sub.levels = sub.render_levels = 2
    return finish(obj, mat)


def tube(points, radius, mat):
    """A smooth rounded stroke through 3D points (smiles, mouths, brows)."""
    curve = bpy.data.curves.new("stroke", "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = radius
    curve.bevel_resolution = 8
    curve.use_fill_caps = True
    spline = curve.splines.new("BEZIER")
    spline.bezier_points.add(len(points) - 1)
    for point, co in zip(spline.bezier_points, points, strict=True):
        point.co = co
        point.handle_left_type = point.handle_right_type = "AUTO"
    obj = bpy.data.objects.new("stroke", curve)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat)
    return obj


def on_surface(center, radii, direction, size, mat, sink=0.35):
    """A flattened blob lying on an ellipsoid's surface (stripes, patches)."""
    d = Vector(direction).normalized()
    t = 1 / math.sqrt(sum((d[i] / radii[i]) ** 2 for i in range(3)))
    point = Vector(center) + t * d
    normal = Vector([(point[i] - center[i]) / radii[i] ** 2 for i in range(3)]).normalized()
    across = Vector((1, 0, 0)) if abs(normal.x) < 0.9 else Vector((0, 0, 1))
    along = normal.cross(across).normalized()
    across = along.cross(normal).normalized()
    rotation = Matrix((across, normal, along)).transposed().to_euler()
    location = point - normal * size[1] * sink
    return sphere(location, size, mat, rotation=rotation)


def eye_highlight(location, size, mat):
    return sphere(location, (size, size * 0.6, size), mat)


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("METAL", "CUDA", "OPTIX", "HIP"):
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
            if any(d.type != "CPU" for d in prefs.devices):
                for d in prefs.devices:
                    d.use = True
                scene.cycles.device = "GPU"
                break
        except TypeError:
            continue
    scene.cycles.samples = 256
    scene.cycles.use_denoising = True
    scene.render.film_transparent = True
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = -0.6
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    world = bpy.data.worlds.new("world")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = hex_rgba("#FFFFFF")
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.6
    return scene


def light(name, location, energy, size, color="#FFFFFF"):
    data = bpy.data.lights.new(name, "AREA")
    data.energy = energy
    data.size = size
    data.color = hex_rgba(color)[:3]
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    direction = Vector((0, 0, 0)) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def bounds():
    lo = Vector((1e9, 1e9, 1e9))
    hi = Vector((-1e9, -1e9, -1e9))
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in bpy.context.scene.objects:
        if obj.type not in {"MESH", "CURVE"}:
            continue
        evaluated = obj.evaluated_get(depsgraph)
        for corner in evaluated.bound_box:
            world = evaluated.matrix_world @ Vector(corner)
            lo = Vector(map(min, lo, world))
            hi = Vector(map(max, hi, world))
    return lo, hi


def studio(scene, size, out):
    light("key", (-3.2, -4.2, 4.2), 700, 4.5, "#FFF6EC")
    light("fill", (4.2, -3.0, 1.6), 160, 5.0, "#EEF4FF")
    light("rim", (0.5, 4.0, 3.2), 420, 3.0)
    lo, hi = bounds()
    center = (lo + hi) / 2
    extent = max(hi.x - lo.x, hi.z - lo.z) * 1.12
    cam_data = bpy.data.cameras.new("camera")
    cam_data.lens = 85
    fov = 2 * math.atan(cam_data.sensor_width / (2 * cam_data.lens))
    distance = extent / 2 / math.tan(fov / 2)
    camera = bpy.data.objects.new("camera", cam_data)
    bpy.context.collection.objects.link(camera)
    tilt = math.radians(6)
    camera.location = (
        center.x,
        center.y - distance * math.cos(tilt),
        center.z + distance * math.sin(tilt),
    )
    camera.rotation_euler = (math.radians(90) - tilt, 0, 0)
    scene.camera = camera
    scene.render.resolution_x = scene.render.resolution_y = size
    scene.render.filepath = str(out)
    bpy.ops.render.render(write_still=True)


def robot():
    shell = material("shell", "#FFFFFF", roughness=0.28, coat=0.8, subsurface=0.04)
    rim = material("rim", "#C9D1E0", roughness=0.35, coat=0.4)
    glass = material("glass", "#141A2B", roughness=0.12, coat=1.0)
    glow = material("glow", "#3CC8FF", emission="#2EB8FF", strength=4.0)
    accent = material("accent", "#FF8A65", roughness=0.35, coat=0.5)
    bulb = material("bulb", "#FFB13D", emission="#FF8A1E", strength=5.0)
    blush = material("blush", "#FF6F8E", emission="#FF6F8E", strength=0.6)
    white = material("highlight", "#FFFFFF", emission="#FFFFFF", strength=3.0)
    rounded_box((0, 0, 0.9), (3.1, 2.4, 2.55), 0.82, shell)
    rounded_box((0, -1.05, 0.88), (2.35, 0.5, 1.55), 0.55, glass)
    for side in (-1, 1):
        sphere((side * 0.48, -1.32, 0.98), (0.2, 0.12, 0.28), glow)
        eye_highlight((side * 0.48 - 0.07, -1.43, 1.1), 0.055, white)
        sphere((side * 0.86, -1.3, 0.55), (0.17, 0.05, 0.09), blush)
        cylinder(
            (side * 1.62, 0, 0.9), 0.42, 0.34, rim, rotation=(0, math.radians(90), 0), bevel=0.1
        )
        cylinder(
            (side * 1.82, 0, 0.9), 0.24, 0.16, accent, rotation=(0, math.radians(90), 0), bevel=0.06
        )
    tube([(-0.26, -1.33, 0.6), (0, -1.36, 0.46), (0.26, -1.33, 0.6)], 0.055, glow)
    cylinder((0, 0, 2.38), 0.08, 0.5, rim)
    sphere((0, 0, 2.72), (0.22, 0.22, 0.22), bulb)


def cat():
    fur = material("fur", "#F6A55A", roughness=0.6, subsurface=0.12)
    stripe = material("stripe", "#DD7430", roughness=0.6, subsurface=0.1)
    cream = material("cream", "#FFF4E6", roughness=0.55, subsurface=0.15)
    pink = material("pink", "#F7A6A8", roughness=0.5, subsurface=0.15)
    nose = material("nose", "#EE6F7E", roughness=0.25, coat=0.6)
    eye = material("eye", "#1C1714", roughness=0.05, coat=1.0)
    iris = material("iris", "#7DBE2E", roughness=0.15, coat=1.0)
    white = material("highlight", "#FFFFFF", emission="#FFFFFF", strength=5.0)
    mouth = material("mouth", "#5B3A2A", roughness=0.5)
    blush = material("blush", "#FF8F8F", roughness=0.6, subsurface=0.2)
    head, radii = (0, 0, 0.75), (1.55, 1.3, 1.3)
    sphere(head, radii, fur)
    for side in (-1, 1):
        tilt = (math.radians(-8), side * math.radians(-24), 0)
        cone((side * 0.92, 0.05, 1.95), 0.62, 1.2, fur, rotation=tilt, tip=0.06, scale=(1, 0.55, 1))
        cone((side * 0.9, -0.24, 1.86), 0.4, 0.82, pink, rotation=tilt, tip=0.04, scale=(1, 0.3, 1))
        sphere((side * 0.42, -1.04, 0.0), (0.42, 0.34, 0.34), cream)
        sphere((side * 0.55, -1.08, 0.86), (0.29, 0.16, 0.33), iris)
        sphere((side * 0.55, -1.2, 0.86), (0.11, 0.06, 0.27), eye)
        eye_highlight((side * 0.55 - 0.1, -1.24, 1.0), 0.07, white)
        sphere((side * 1.0, -0.82, 0.32), (0.26, 0.1, 0.15), blush)
        for dz in (0.05, 0.3):
            on_surface(head, radii, (side, -0.62, dz), (0.06, 0.05, 0.22), stripe, sink=0.5)
    for dx, length in ((-0.22, 0.24), (0, 0.32), (0.22, 0.24)):
        on_surface(head, radii, (dx, -1.0, 0.8), (0.06, 0.05, length), stripe, sink=0.5)
    for side in (-1, 1):
        for dz, end in ((0.12, 0.22), (0.0, -0.02)):
            tube(
                [(side * 0.62, -1.18, dz), (side * 1.0, -1.12, dz + 0.03), (side * 1.4, -1.0, end)],
                0.016,
                mouth,
            )
    sphere((0, -0.98, -0.32), (0.55, 0.36, 0.3), cream)
    sphere((0, -1.33, 0.24), (0.17, 0.11, 0.12), nose)
    tube([(0, -1.37, 0.12), (0, -1.38, 0.0)], 0.035, mouth)
    tube([(-0.3, -1.33, 0.02), (-0.15, -1.4, -0.12), (0, -1.38, 0.0)], 0.035, mouth)
    tube([(0.3, -1.33, 0.02), (0.15, -1.4, -0.12), (0, -1.38, 0.0)], 0.035, mouth)


def owl():
    body = material("body", "#A9744F", roughness=0.6, subsurface=0.08)
    wing = material("wing", "#7C5136", roughness=0.6, subsurface=0.06)
    belly = material("belly", "#F5E4C8", roughness=0.6, subsurface=0.12)
    disc = material("disc", "#FBF0DE", roughness=0.55, subsurface=0.12)
    iris = material("iris", "#FFB833", roughness=0.15, coat=1.0)
    pupil = material("pupil", "#16110E", roughness=0.05, coat=1.0)
    beak = material("beak", "#F29A2E", roughness=0.3, coat=0.6)
    white = material("highlight", "#FFFFFF", emission="#FFFFFF", strength=5.0)
    mark = material("mark", "#C9A076", roughness=0.6)
    sphere((0, 0, 0.55), (1.45, 1.25, 1.75), body)
    sphere((0, -0.62, 0.12), (1.0, 0.8, 1.12), belly)
    for side in (-1, 1):
        sphere((side * 1.22, 0.05, 0.25), (0.42, 0.9, 1.15), wing, rotation=(0, side * 0.2, 0))
        cone(
            (side * 0.9, 0.0, 2.2),
            0.42,
            0.95,
            wing,
            rotation=(0, side * math.radians(-28), 0),
            tip=0.04,
            scale=(1, 0.6, 1),
        )
        sphere((side * 0.55, -1.0, 1.25), (0.62, 0.3, 0.6), disc)
        sphere((side * 0.55, -1.22, 1.25), (0.44, 0.16, 0.44), iris)
        sphere((side * 0.55, -1.36, 1.25), (0.24, 0.08, 0.24), pupil)
        eye_highlight((side * 0.55 - 0.13, -1.44, 1.38), 0.08, white)
        for k in (-1, 0, 1):
            sphere((side * 0.32 + k * 0.12, -0.62, -1.14), (0.09, 0.14, 0.08), beak)
        tube(
            [(side * 0.95, -1.08, 1.82), (side * 0.55, -1.25, 1.95), (side * 0.12, -1.22, 1.72)],
            0.06,
            wing,
        )
    cone(
        (0, -1.38, 0.82),
        0.17,
        0.45,
        beak,
        rotation=(math.radians(180), 0, 0),
        tip=0.01,
        scale=(1, 0.7, 1),
    )
    for row, (z, xs) in enumerate(
        ((0.2, (-0.3, 0, 0.3)), (-0.15, (-0.45, -0.15, 0.15, 0.45)), (-0.5, (-0.3, 0, 0.3)))
    ):
        for x in xs:
            tube(
                [
                    (x - 0.11, -1.4 + 0.05 * row, z + 0.04),
                    (x, -1.43 + 0.05 * row, z - 0.05),
                    (x + 0.11, -1.4 + 0.05 * row, z + 0.04),
                ],
                0.03,
                mark,
            )


CHARACTERS = {"robot": robot, "cat": cat, "owl": owl}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", default=list(CHARACTERS))
    parser.add_argument("--out", type=Path, default=ROOT / "figures/assets")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--samples", type=int, default=256)
    args = parser.parse_args()
    for name in args.names:
        scene = reset()
        scene.cycles.samples = args.samples
        CHARACTERS[name]()
        studio(scene, args.size, args.out / f"{name}.png")


if __name__ == "__main__":
    main()
