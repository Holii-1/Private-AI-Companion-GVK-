#!/usr/bin/env python3
"""
GVK Private AI - 3D Dock Chassis & Studio Viewer Generator (hardware/generate_3d_dock.py)

Generates printable 3D meshes and an interactive Three.js studio viewer for the
GVK-DOCK MK.I slanted desktop hardware console.

Key Responsibilities:
  - Construct the slanted console chassis, vertical Blu-ray drive bay, 3-slot
    cartridge bay, status LEDs, and lower kill-switch control apron.
  - Export watertight binary STL (`hardware/gvk_dock_mk1.stl`) for 3D slicers
    and colored OBJ/MTL (`hardware/gvk_dock_mk1.obj`, `.mtl`) for Blender/QuickLook.
  - Generate an interactive Three.js HTML viewer (`hardware/gvk_dock_viewer.html`).

Usage:
  python3 hardware/generate_3d_dock.py
"""

import os
import math
import struct


# =====================================================================
# 3D MESH BUILDER & EXPORTERS
# =====================================================================
class MeshBuilder:
    """Builds 3D CAD geometry with rotated local frames and exports OBJ/MTL and STL."""

    def __init__(self):
        """Initialize empty vertex, face, and material registries."""
        self.vertices = []
        self.faces = []  # (material, [v1, v2, v3, ...]) 1-based indices
        self.materials = {}
        self.current_mat = "MatteBlack"

    def add_material(self, name, kd, ks=(0.2, 0.2, 0.2), ns=20.0):
        """Register a named Phong material for MTL export.

        Args:
            name: Material identifier referenced by face groups.
            kd: Diffuse RGB color tuple in `[0.0, 1.0]`.
            ks: Specular RGB color tuple in `[0.0, 1.0]`.
            ns: Specular shininess exponent.
        """
        self.materials[name] = {"kd": kd, "ks": ks, "ns": ns}

    def use_mat(self, name):
        """Set the active material for subsequent faces."""
        self.current_mat = name

    def v(self, x, y, z):
        """Append a 3D vertex and return its 1-based OBJ index."""
        self.vertices.append((float(x), float(y), float(z)))
        return len(self.vertices)

    def quad(self, i1, i2, i3, i4, mat=None):
        """Append a quad face using 1-based vertex indices."""
        m = mat or self.current_mat
        self.faces.append((m, [i1, i2, i3, i4]))

    def tri(self, i1, i2, i3, mat=None):
        """Append a triangle face using 1-based vertex indices."""
        m = mat or self.current_mat
        self.faces.append((m, [i1, i2, i3]))

    def add_box(self, cx, cy, cz, sx, sy, sz, rx_deg=0.0, ry_deg=0.0, rz_deg=0.0, mat=None):
        """Add an oriented rectangular prism centered at `(cx, cy, cz)`.

        Uses the Z-up CAD/STL coordinate convention:
        X left (-) to right (+), Y front (-) to back (+), Z bottom (0) to top (+).

        Args:
            cx: Center X coordinate in millimeters.
            cy: Center Y coordinate in millimeters.
            cz: Center Z coordinate in millimeters.
            sx: Box width along local X in millimeters.
            sy: Box depth along local Y in millimeters.
            sz: Box height along local Z in millimeters.
            rx_deg: Rotation around X axis in degrees.
            ry_deg: Rotation around Y axis in degrees.
            rz_deg: Rotation around Z axis in degrees.
            mat: Optional material name override.
        """
        if mat:
            self.use_mat(mat)
        hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
        local_pts = [
            (-hx, -hy, -hz),
            ( hx, -hy, -hz),
            ( hx,  hy, -hz),
            (-hx,  hy, -hz),
            (-hx, -hy,  hz),
            ( hx, -hy,  hz),
            ( hx,  hy,  hz),
            (-hx,  hy,  hz),
        ]
        rx = math.radians(rx_deg)
        ry = math.radians(ry_deg)
        rz = math.radians(rz_deg)
        cx_r, sx_r = math.cos(rx), math.sin(rx)
        cy_r, sy_r = math.cos(ry), math.sin(ry)
        cz_r, sz_r = math.cos(rz), math.sin(rz)

        idx = []
        for lx, ly, lz in local_pts:
            # Rotate X
            y1 = ly * cx_r - lz * sx_r
            z1 = ly * sx_r + lz * cx_r
            x1 = lx
            # Rotate Y
            x2 = x1 * cy_r + z1 * sy_r
            z2 = -x1 * sy_r + z1 * cy_r
            y2 = y1
            # Rotate Z
            x3 = x2 * cz_r - y2 * sz_r
            y3 = x2 * sz_r + y2 * cz_r
            z3 = z2
            idx.append(self.v(cx + x3, cy + y3, cz + z3))

        # Bottom, Top, Front, Back, Left, Right
        self.quad(idx[3], idx[2], idx[1], idx[0])
        self.quad(idx[4], idx[5], idx[6], idx[7])
        self.quad(idx[0], idx[1], idx[5], idx[4])
        self.quad(idx[2], idx[3], idx[7], idx[6])
        self.quad(idx[3], idx[0], idx[4], idx[7])
        self.quad(idx[1], idx[2], idx[6], idx[5])

    def add_cylinder(self, cx, cy, cz, radius, length, rx_deg=0.0, segments=20, mat=None):
        """Add a closed cylinder along local Z, rotated by `rx_deg` around X."""
        if mat:
            self.use_mat(mat)
        rx = math.radians(rx_deg)
        c_r, s_r = math.cos(rx), math.sin(rx)
        hz = length / 2.0

        def xform(lx, ly, lz):
            y1 = ly * c_r - lz * s_r
            z1 = ly * s_r + lz * c_r
            return self.v(cx + lx, cy + y1, cz + z1)

        bot_center = xform(0, 0, -hz)
        top_center = xform(0, 0, hz)
        bot_ring = []
        top_ring = []
        for i in range(segments):
            ang = 2.0 * math.pi * i / segments
            px = radius * math.cos(ang)
            py = radius * math.sin(ang)
            bot_ring.append(xform(px, py, -hz))
            top_ring.append(xform(px, py, hz))

        for i in range(segments):
            ni = (i + 1) % segments
            self.quad(bot_ring[i], bot_ring[ni], top_ring[ni], top_ring[i])
            self.tri(bot_center, bot_ring[ni], bot_ring[i])
            self.tri(top_center, top_ring[i], top_ring[ni])

    def add_profile_extrusion(self, x_min, x_max, yz_polygon, mat=None):
        """Extrude a 2D (Y, Z) polygon along the X axis from `x_min` to `x_max`.

        Args:
            x_min: Left X boundary in millimeters.
            x_max: Right X boundary in millimeters.
            yz_polygon: Sequence of `(y, z)` vertices ordered counter-clockwise when viewed from `+X`.
            mat: Optional material name override.
        """
        if mat:
            self.use_mat(mat)
        n = len(yz_polygon)
        left_v = [self.v(x_min, y, z) for (y, z) in yz_polygon]
        right_v = [self.v(x_max, y, z) for (y, z) in yz_polygon]

        # Side perimeter quads
        for i in range(n):
            ni = (i + 1) % n
            self.quad(left_v[i], right_v[i], right_v[ni], left_v[ni])

        # Triangulate convex/simple polygon fan from vertex 0 for left and right caps
        for i in range(1, n - 1):
            self.tri(left_v[0], left_v[i + 1], left_v[i])
            self.tri(right_v[0], right_v[i], right_v[i + 1])

    def export_obj(self, obj_path, mtl_path):
        """Write Wavefront OBJ geometry (Y-up) and companion MTL material library.

        Args:
            obj_path: Output `.obj` file path.
            mtl_path: Output `.mtl` file path.
        """
        mtl_name = os.path.basename(mtl_path)
        with open(obj_path, "w", encoding="utf-8") as f:
            f.write("# GVK Private AI - GVK-DOCK MK.I Slanted Console 3D Model\n")
            f.write(f"mtllib {mtl_name}\n\n")
            # Write vertices (converting Z-up CAD to Y-up for QuickLook/Three.js friendliness: X, Z, -Y)
            for vx, vy, vz in self.vertices:
                f.write(f"v {vx:.3f} {vz:.3f} {-vy:.3f}\n")
            f.write("\n")
            cur_m = None
            for m, idxs in self.faces:
                if m != cur_m:
                    cur_m = m
                    f.write(f"usemtl {m}\n")
                f.write("f " + " ".join(str(i) for i in idxs) + "\n")

        with open(mtl_path, "w", encoding="utf-8") as f:
            f.write("# GVK-DOCK // MK.I Materials\n")
            for name, p in self.materials.items():
                f.write(f"\nnewmtl {name}\n")
                f.write(f"Kd {p['kd'][0]:.3f} {p['kd'][1]:.3f} {p['kd'][2]:.3f}\n")
                f.write(f"Ks {p['ks'][0]:.3f} {p['ks'][1]:.3f} {p['ks'][2]:.3f}\n")
                f.write(f"Ns {p['ns']:.1f}\n")
                f.write("illum 2\n")

    def export_binary_stl(self, stl_path, filter_materials=None):
        """Write a binary STL mesh (Z-up, millimeter units) ready for 3D slicers.

        Args:
            stl_path: Output `.stl` file path.
            filter_materials: Optional set/collection of material names to include.
        """
        tris = []
        for m, idxs in self.faces:
            if filter_materials and m not in filter_materials:
                continue
            if len(idxs) == 3:
                tris.append((idxs[0] - 1, idxs[1] - 1, idxs[2] - 1))
            elif len(idxs) == 4:
                tris.append((idxs[0] - 1, idxs[1] - 1, idxs[2] - 1))
                tris.append((idxs[0] - 1, idxs[2] - 1, idxs[3] - 1))

        with open(stl_path, "wb") as f:
            header = b"GVK-DOCK MK.I 3D Printable Console (mm units)".ljust(80, b" ")
            f.write(header)
            f.write(struct.pack("<I", len(tris)))
            for i1, i2, i3 in tris:
                v1 = self.vertices[i1]
                v2 = self.vertices[i2]
                v3 = self.vertices[i3]
                ux, uy, uz = v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]
                vx, vy, vz = v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]
                nx = uy * vz - uz * vy
                ny = uz * vx - ux * vz
                nz = ux * vy - uy * vx
                length = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
                nx, ny, nz = nx / length, ny / length, nz / length
                f.write(struct.pack("<12fH", nx, ny, nz, *v1, *v2, *v3, 0))


# =====================================================================
# CONSOLE GEOMETRY GENERATION
# =====================================================================
def generate_all(output_dir):
    """Build the complete GVK-DOCK MK.I mesh and export OBJ, MTL, and STL files.

    Args:
        output_dir: Directory where generated 3D model files are written.
    """
    os.makedirs(output_dir, exist_ok=True)
    m = MeshBuilder()

    # Materials matching the photo
    m.add_material("CarbonBody", kd=(0.10, 0.11, 0.12), ks=(0.35, 0.35, 0.38), ns=35.0)
    m.add_material("MatteBezel", kd=(0.07, 0.07, 0.08), ks=(0.20, 0.20, 0.22), ns=20.0)
    m.add_material("DarkBolt", kd=(0.04, 0.04, 0.05), ks=(0.50, 0.50, 0.55), ns=60.0)
    m.add_material("SilverPlate", kd=(0.78, 0.80, 0.83), ks=(0.90, 0.90, 0.95), ns=80.0)
    m.add_material("ChromeMetal", kd=(0.85, 0.87, 0.90), ks=(0.98, 0.98, 0.98), ns=110.0)
    m.add_material("RedMissileCover", kd=(0.82, 0.06, 0.06), ks=(0.70, 0.30, 0.30), ns=50.0)
    m.add_material("LED_Green", kd=(0.10, 0.95, 0.35), ks=(0.90, 0.90, 0.90), ns=60.0)
    m.add_material("LED_Red", kd=(0.98, 0.15, 0.15), ks=(0.90, 0.90, 0.90), ns=60.0)
    m.add_material("CaseBlack", kd=(0.13, 0.14, 0.16), ks=(0.25, 0.25, 0.28), ns=25.0)
    m.add_material("BadgeCyan", kd=(0.05, 0.78, 0.92), ks=(0.6, 0.6, 0.6), ns=40.0)
    m.add_material("BadgeYellow", kd=(0.95, 0.78, 0.10), ks=(0.6, 0.6, 0.6), ns=40.0)
    m.add_material("BadgeViolet", kd=(0.68, 0.28, 0.88), ks=(0.6, 0.6, 0.6), ns=40.0)
    m.add_material("RubberFoot", kd=(0.03, 0.03, 0.03), ks=(0.05, 0.05, 0.05), ns=5.0)

    # =====================================================================
    # GEOMETRY OF THE SLANTED CONSOLE (PHOTO-EXACT PROFILE)
    # =====================================================================
    # Total Width X: 240 mm (-120 to +120)
    # Side profile in (Y, Z):
    #   Front bottom lip: (Y = -90, Z = 0) -> (Y = -90, Z = 16)
    #   Lower control apron (sloped ~28 deg): (Y = -90, Z = 16) -> (Y = -42, Z = 44)
    #   Upper main face (sloped ~68 deg, i.e. 22 deg from vertical):
    #     from (Y = -42, Z = 44) -> (Y = +22, Z = 202)
    #   Top flat roof: (Y = +22, Z = 202) -> (Y = +85, Z = 202)
    #   Back vertical wall: (Y = +85, Z = 202) -> (Y = +85, Z = 0)

    profile_main = [
        (-90.0, 0.0),
        ( 85.0, 0.0),
        ( 85.0, 202.0),
        ( 22.0, 202.0),
        (-42.0, 44.0),
        (-90.0, 16.0),
    ]
    m.add_profile_extrusion(-120.0, 120.0, profile_main, mat="CarbonBody")

    # Side raised carbon-fiber trim panels (left and right flanks, 3mm thick)
    profile_flank = [
        (-91.0, 2.0),
        ( 86.0, 2.0),
        ( 86.0, 204.0),
        ( 21.0, 204.0),
        (-43.0, 45.0),
        (-91.0, 17.0),
    ]
    m.add_profile_extrusion(-123.0, -120.0, profile_flank, mat="MatteBezel")
    m.add_profile_extrusion( 120.0,  123.0, profile_flank, mat="MatteBezel")

    # Angle of upper faceplate:
    # dy = 22 - (-42) = 64, dz = 202 - 44 = 158
    # Normal tilts forward; tilt angle from vertical is atan2(64, 158) = 22.05 deg
    upper_tilt_deg = -22.05  # rotation around X so local Z points out of the upper face, wait:
    # Let's align boxes on the upper faceplate so their local Y is normal or local Z is along the face:
    # If a box has height along Z and thickness along Y, rotating around X by -22.05 deg tilts its top (+Z) toward +Y (backwards)!
    face_rot_x = -22.05

    def upper_face_pt(u_x, frac_up, normal_offset=0.0):
        """Returns (x, y, z) on the upper sloped faceplate for fraction frac_up in [0..1]."""
        y0, z0 = -42.0, 44.0
        y1, z1 = 22.0, 202.0
        py = y0 + (y1 - y0) * frac_up
        pz = z0 + (z1 - z0) * frac_up
        # Outward unit normal of upper face: (-dz, dy) / len = (0, -158/170.4, 64/170.4) = (0, -0.927, 0.375)
        ny, nz = -0.927, 0.375
        return (u_x, py + ny * normal_offset, pz + nz * normal_offset)

    # =====================================================================
    # 1. VERTICAL SLIM BLU-RAY DRIVE BAY (LEFT SIDE OF UPPER FACEPLATE)
    # =====================================================================
    # Outer bezel frame around the vertical Blu-ray drive
    bx, by, bz = upper_face_pt(-62.0, 0.48, normal_offset=2.0)
    m.add_box(bx, by, bz, 62.0, 8.0, 142.0, rx_deg=face_rot_x, mat="MatteBezel")

    # Inner vertical Blu-ray drive faceplate
    dx, dy, dz = upper_face_pt(-62.0, 0.48, normal_offset=5.0)
    m.add_box(dx, dy, dz, 44.0, 6.0, 128.0, rx_deg=face_rot_x, mat="CarbonBody")

    # Disc tray slot seam & raised lip
    tx, ty, tz = upper_face_pt(-65.0, 0.48, normal_offset=7.5)
    m.add_box(tx, ty, tz, 18.0, 4.0, 116.0, rx_deg=face_rot_x, mat="MatteBezel")

    # Eject button on vertical drive
    ex, ey, ez = upper_face_pt(-48.0, 0.66, normal_offset=7.5)
    m.add_box(ex, ey, ez, 7.0, 4.0, 14.0, rx_deg=face_rot_x, mat="MatteBezel")

    # 6 Hex Screws on Drive Bezel
    for sx_off in [-87.0, -37.0]:
        for f_up in [0.10, 0.48, 0.86]:
            px, py, pz = upper_face_pt(sx_off, f_up, normal_offset=6.5)
            m.add_cylinder(px, py, pz, 2.2, 3.0, rx_deg=90 + face_rot_x, segments=6, mat="DarkBolt")

    # =====================================================================
    # 2. SIDE-BY-SIDE CARTRIDGE BAY & 3 PROTRUDING CASES (RIGHT SIDE)
    # =====================================================================
    # Cartridge housing collar/frame
    rx, ry, rz = upper_face_pt(48.0, 0.56, normal_offset=1.5)
    m.add_box(rx, ry, rz, 118.0, 10.0, 105.0, rx_deg=face_rot_x, mat="MatteBezel")

    # 3 Blu-ray Cartridge Cases protruding out of the angled bay side-by-side!
    # Each case spine is 28mm wide (heavy-duty collector's slipcase), protruding 52mm out
    case_specs = [
        ( 12.0, "BadgeCyan"),    # FRIEND 01
        ( 46.0, "BadgeYellow"),  # FRIEND 02
        ( 80.0, "BadgeViolet"),  # FRIEND 03
    ]
    for cx_pos, badge_mat in case_specs:
        # Main protruding black cartridge box
        px, py, pz = upper_face_pt(cx_pos, 0.62, normal_offset=28.0)
        m.add_box(px, py, pz, 29.0, 62.0, 108.0, rx_deg=face_rot_x, mat="CaseBlack")
        # Colored circle indicator on the spine
        bx, by, bz = upper_face_pt(cx_pos, 0.72, normal_offset=59.2)
        m.add_cylinder(bx, by, bz, 4.5, 1.5, rx_deg=90 + face_rot_x, segments=16, mat=badge_mat)
        # Bottom color accent bar on spine
        lx, ly, lz = upper_face_pt(cx_pos, 0.36, normal_offset=59.2)
        m.add_box(lx, ly, lz, 18.0, 1.5, 2.5, rx_deg=face_rot_x, mat=badge_mat)

    # =====================================================================
    # 3. STATUS LED BEZEL PLATE BELOW CARTRIDGES (SUBJ-01 / 02 / 03 / VAULT)
    # =====================================================================
    lx, ly, lz = upper_face_pt(48.0, 0.13, normal_offset=3.5)
    m.add_box(lx, ly, lz, 92.0, 5.0, 22.0, rx_deg=face_rot_x, mat="MatteBezel")

    led_mats = ["LED_Green", "LED_Green", "LED_Red", "LED_Red"]
    for i, lmat in enumerate(led_mats):
        led_x = 15.0 + i * 22.0
        px, py, pz = upper_face_pt(led_x, 0.15, normal_offset=6.5)
        m.add_cylinder(px, py, pz, 2.8, 2.5, rx_deg=90 + face_rot_x, segments=14, mat=lmat)

    # =====================================================================
    # 4. LOWER SLOPED CONTROL APRON (STATUS LED, SILVER ID PLATE, KILL-SWITCH)
    # =====================================================================
    # Lower apron goes from (Y = -90, Z = 16) to (Y = -42, Z = 44)
    # dy = 48, dz = 28 -> slope angle = atan2(28, 48) = 30.25 deg from horizontal
    # So rotation around X from horizontal top face is -(90 - 30.25) = -59.75 deg
    apron_rot_x = -59.75

    def apron_pt(u_x, frac_up, normal_offset=0.0):
        y0, z0 = -90.0, 16.0
        y1, z1 = -42.0, 44.0
        py = y0 + (y1 - y0) * frac_up
        pz = z0 + (z1 - z0) * frac_up
        # Outward unit normal: (-28, 48) / 55.57 = (0, -0.504, 0.864)
        return (u_x, py - 0.504 * normal_offset, pz + 0.864 * normal_offset)

    # Green Status Pill LED on bottom-left ([ STATUS: ONLINE ])
    gx, gy, gz = apron_pt(-82.0, 0.48, normal_offset=2.0)
    m.add_box(gx, gy, gz, 5.0, 3.0, 11.0, rx_deg=apron_rot_x, mat="LED_Green")

    # Brushed Aluminum Station ID Plate in the center
    sx, sy, sz = apron_pt(-8.0, 0.48, normal_offset=2.0)
    m.add_box(sx, sy, sz, 66.0, 2.5, 22.0, rx_deg=apron_rot_x, mat="SilverPlate")
    # 4 rivets on the silver plate
    for rx_off in [-37.0, 21.0]:
        for rf_up in [0.28, 0.68]:
            px, py, pz = apron_pt(rx_off, rf_up, normal_offset=3.5)
            m.add_cylinder(px, py, pz, 1.4, 1.5, rx_deg=90 + apron_rot_x, segments=8, mat="DarkBolt")

    # Aviation Kill-Switch Toggle + Raised Red Flip Cover on bottom-right
    kx, ky, kz = apron_pt(76.0, 0.42, normal_offset=2.5)
    # Chrome Hex Base Nut
    m.add_cylinder(kx, ky, kz, 9.0, 5.0, rx_deg=90 + apron_rot_x, segments=6, mat="ChromeMetal")
    # Chrome Toggle Lever
    lx, ly, lz = apron_pt(76.0, 0.42, normal_offset=12.0)
    m.add_cylinder(lx, ly, lz, 2.8, 18.0, rx_deg=75 + apron_rot_x, segments=12, mat="ChromeMetal")
    # Red Flip Guard Cover (Open / Raised upright position as in photo)
    cx, cy, cz = apron_pt(76.0, 0.82, normal_offset=18.0)
    m.add_box(cx, cy, cz, 20.0, 14.0, 28.0, rx_deg=apron_rot_x + 45.0, mat="RedMissileCover")

    # Perimeter Hex Bolts on Main Faceplate & Lower Apron
    for bx_pos in [-112.0, 112.0]:
        for f_u in [0.06, 0.92]:
            px, py, pz = upper_face_pt(bx_pos, f_u, normal_offset=1.5)
            m.add_cylinder(px, py, pz, 2.5, 2.5, rx_deg=90 + face_rot_x, segments=6, mat="DarkBolt")
        for f_a in [0.22, 0.78]:
            px, py, pz = apron_pt(bx_pos, f_a, normal_offset=1.5)
            m.add_cylinder(px, py, pz, 2.5, 2.5, rx_deg=90 + apron_rot_x, segments=6, mat="DarkBolt")

    # 4 Rubber Feet under base
    for fx in [-105.0, 105.0]:
        for fy in [-75.0, 70.0]:
            m.add_box(fx, fy, -2.0, 16.0, 16.0, 4.0, mat="RubberFoot")

    # Write files
    obj_path = os.path.join(output_dir, "gvk_dock_mk1.obj")
    mtl_path = os.path.join(output_dir, "gvk_dock_mk1.mtl")
    stl_path = os.path.join(output_dir, "gvk_dock_mk1.stl")

    m.export_obj(obj_path, mtl_path)
    m.export_binary_stl(stl_path)
    print(f"[OK] Exported OBJ model: {obj_path} ({len(m.vertices)} vertices, {len(m.faces)} faces)")
    print(f"[OK] Exported binary STL: {stl_path}")


# =====================================================================
# INTERACTIVE THREE.JS HTML VIEWER GENERATOR
# =====================================================================
def generate_photoreal_viewer_html(output_path):
    """Write the standalone interactive Three.js 3D viewer HTML file.

    Args:
        output_path: Destination `.html` file path.
    """
    html = """<!DOCTYPE html>
<html lang="uk">
<head>
  <meta charset="UTF-8">
  <title>GVK NEURAL ARCHIVE // UNIT 01 — 3D Консоль</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: radial-gradient(circle at 50% 35%, #1c2026 0%, #0b0d10 100%);
      color: #e2e8f0;
      font-family: "JetBrains Mono", -apple-system, BlinkMacSystemFont, monospace;
      overflow: hidden;
      height: 100vh;
      user-select: none;
    }
    #canvas-container { width: 100vw; height: 100vh; }
    #hud {
      position: absolute;
      top: 20px;
      left: 20px;
      background: rgba(14, 16, 20, 0.90);
      border: 1px solid #2d3748;
      border-left: 4px solid #10b981;
      border-radius: 6px;
      padding: 16px 20px;
      width: 360px;
      backdrop-filter: blur(10px);
      box-shadow: 0 12px 32px rgba(0,0,0,0.65);
    }
    h1 {
      font-size: 13px;
      letter-spacing: 1.5px;
      color: #f8fafc;
      margin-bottom: 4px;
    }
    .rev { color: #10b981; font-size: 11px; float: right; }
    .sub {
      font-size: 11px;
      color: #94a3b8;
      margin-bottom: 12px;
      line-height: 1.4;
    }
    .metric {
      display: flex;
      justify-content: space-between;
      font-size: 11px;
      padding: 4px 0;
      border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    .metric span:first-child { color: #94a3b8; }
    .metric span:last-child { color: #e2e8f0; font-weight: 600; }
    .btns {
      margin-top: 14px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    button {
      background: #1e2530;
      border: 1px solid #334155;
      color: #e2e8f0;
      padding: 7px 10px;
      border-radius: 4px;
      font-family: inherit;
      font-size: 10px;
      cursor: pointer;
      transition: all 0.18s;
    }
    button:hover {
      background: #2dd4bf22;
      border-color: #10b981;
      color: #10b981;
    }
    button.danger:hover {
      background: #ef444422;
      border-color: #ef4444;
      color: #ef4444;
    }
    #hint {
      position: absolute;
      bottom: 18px;
      left: 50%;
      transform: translateX(-50%);
      background: rgba(14, 16, 20, 0.8);
      border: 1px solid #2d3748;
      padding: 8px 18px;
      border-radius: 20px;
      font-size: 11px;
      color: #94a3b8;
      pointer-events: none;
    }
  </style>
</head>
<body>
  <div id="hud">
    <h1>[ GVK NEURAL ARCHIVE // UNIT 01 ] <span class="rev">[REV 2.6]</span></h1>
    <div class="sub">Точна 3D-реконструкція лабораторної консолі за твоїм референс-фото.</div>
    <div class="metric"><span>Габарити основи:</span><span>240 × 175 мм</span></div>
    <div class="metric"><span>Висота консолі:</span><span>202 мм (кут 68°)</span></div>
    <div class="metric"><span>Лівий відсік:</span><span>Вертикальний BD-ROM</span></div>
    <div class="metric"><span>Правий відсік:</span><span>Касетниця (3× бокси)</span></div>
    <div class="metric"><span>Нижня панель:</span><span>Шильдик + Kill-Switch</span></div>
    <div class="btns">
      <button id="btn-eject">Вийняти/Вставити Картридж 01</button>
      <button id="btn-kill" class="danger">Клацнути Kill-Switch</button>
      <button id="btn-wire">Сітка / Корпус</button>
      <button id="btn-cam">Ракурс як на фото</button>
    </div>
  </div>

  <div id="hint">🖱️ ЛКМ — обертати 3D модель | Колесо — зум | ПКМ — панорама</div>
  <div id="canvas-container"></div>

  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>

  <script>
    const container = document.getElementById('canvas-container');
    const scene = new THREE.Scene();

    const camera = new THREE.PerspectiveCamera(36, window.innerWidth / window.innerHeight, 1, 3000);
    // Exact camera angle from the uploaded reference photo!
    camera.position.set(-220, 215, 340);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.target.set(0, 95, 0);

    // Studio Lighting
    scene.add(new THREE.AmbientLight(0xdbeafe, 0.55));

    const keyLight = new THREE.DirectionalLight(0xfffbeb, 1.3);
    keyLight.position.set(-180, 380, 260);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.set(2048, 2048);
    scene.add(keyLight);

    const rimLight = new THREE.DirectionalLight(0x60a5fa, 0.75);
    rimLight.position.set(260, 220, -240);
    scene.add(rimLight);

    // Warm Wooden Desk Surface (like in the photo)
    const deskGeo = new THREE.PlaneGeometry(1400, 900);
    const deskMat = new THREE.MeshStandardMaterial({ color: 0x3d2817, roughness: 0.75, metalness: 0.05 });
    const desk = new THREE.Mesh(deskGeo, deskMat);
    desk.rotation.x = -Math.PI / 2;
    desk.position.y = -2;
    desk.receiveShadow = true;
    scene.add(desk);

    // Helper: Procedural Carbon Fiber / Brushed Texture via Canvas
    function makeCarbonTexture() {
      const c = document.createElement('canvas');
      c.width = 64; c.height = 64;
      const ctx = c.getContext('2d');
      ctx.fillStyle = '#141619';
      ctx.fillRect(0, 0, 64, 64);
      ctx.fillStyle = '#1c1f24';
      for (let y = 0; y < 64; y += 8) {
        for (let x = 0; x < 64; x += 8) {
          ctx.fillRect(x + ((y/8)%2)*4, y, 4, 4);
        }
      }
      const tex = new THREE.CanvasTexture(c);
      tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
      tex.repeat.set(24, 24);
      return tex;
    }

    function makeSpineTexture(numStr, dotColor, friendLabel) {
      const c = document.createElement('canvas');
      c.width = 256; c.height = 1024;
      const ctx = c.getContext('2d');
      ctx.fillStyle = '#181a1f';
      ctx.fillRect(0, 0, 256, 1024);

      ctx.strokeStyle = '#2d333b';
      ctx.lineWidth = 6;
      ctx.strokeRect(10, 10, 236, 1004);

      ctx.fillStyle = '#cbd5e1';
      ctx.font = 'bold 42px monospace';
      ctx.textAlign = 'center';
      ctx.fillText('GVK', 128, 95);

      ctx.strokeStyle = '#64748b';
      ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(45, 120); ctx.lineTo(211, 120); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(45, 235); ctx.lineTo(211, 235); ctx.stroke();

      ctx.fillStyle = '#f8fafc';
      ctx.font = 'bold 74px monospace';
      ctx.fillText(numStr, 128, 205);

      // Colored circle
      ctx.fillStyle = dotColor;
      ctx.beginPath();
      ctx.arc(128, 320, 34, 0, Math.PI * 2);
      ctx.fill();

      // Vertical FRIEND label
      ctx.save();
      ctx.translate(128, 660);
      ctx.rotate(Math.PI / 2);
      ctx.fillStyle = '#e2e8f0';
      ctx.font = 'bold 48px monospace';
      ctx.fillText(friendLabel, 0, 16);
      ctx.restore();

      // Bottom color bar
      ctx.fillStyle = dotColor;
      ctx.fillRect(50, 940, 156, 14);

      return new THREE.CanvasTexture(c);
    }

    function makeSilverPlateTexture() {
      const c = document.createElement('canvas');
      c.width = 512; c.height = 180;
      const ctx = c.getContext('2d');
      const grad = ctx.createLinearGradient(0, 0, 512, 180);
      grad.addColorStop(0, '#d1d5db');
      grad.addColorStop(0.5, '#f3f4f6');
      grad.addColorStop(1, '#9ca3af');
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, 512, 180);
      ctx.strokeStyle = '#374151';
      ctx.lineWidth = 6;
      ctx.strokeRect(6, 6, 500, 168);

      ctx.fillStyle = '#111827';
      ctx.font = 'bold 30px monospace';
      ctx.textAlign = 'center';
      ctx.fillText('STATION ID: 01 OF 03', 256, 78);
      ctx.fillText('// GVK VAULT MK.I', 256, 128);
      return new THREE.CanvasTexture(c);
    }

    function makeHeaderTexture() {
      const c = document.createElement('canvas');
      c.width = 1024; c.height = 96;
      const ctx = c.getContext('2d');
      ctx.fillStyle = '#141619';
      ctx.fillRect(0, 0, 1024, 96);
      ctx.fillStyle = '#cbd5e1';
      ctx.font = 'bold 34px monospace';
      ctx.fillText('[ GVK NEURAL ARCHIVE // UNIT 01 ]', 30, 60);
      ctx.fillText('[REV 2.6]', 810, 60);
      return new THREE.CanvasTexture(c);
    }

    const carbonTex = makeCarbonTexture();
    const matCarbon = new THREE.MeshStandardMaterial({
      color: 0x1a1d22,
      map: carbonTex,
      roughness: 0.55,
      metalness: 0.25
    });
    const matDarkTrim = new THREE.MeshStandardMaterial({ color: 0x111317, roughness: 0.45, metalness: 0.35 });
    const matBolt = new THREE.MeshStandardMaterial({ color: 0x090a0c, roughness: 0.3, metalness: 0.8 });
    const matChrome = new THREE.MeshStandardMaterial({ color: 0xe2e8f0, roughness: 0.12, metalness: 0.95 });
    const matRedCover = new THREE.MeshStandardMaterial({ color: 0xdc2626, roughness: 0.35, metalness: 0.2 });
    const matGreenLED = new THREE.MeshBasicMaterial({ color: 0x22c55e });
    const matRedLED = new THREE.MeshBasicMaterial({ color: 0xef4444 });

    const root = new THREE.Group();
    scene.add(root);

    // 1. EXTRUDED SLANTED CONSOLE BODY
    // In Three.js: X is Left-Right, Y is Up, Z is Front(+)/Back(-)
    // Let's build the 2D side profile in (Z, Y) and extrude along X:
    const shape = new THREE.Shape();
    shape.moveTo(-85, 0);     // Back bottom
    shape.lineTo( 90, 0);     // Front bottom
    shape.lineTo( 90, 16);    // Front lip
    shape.lineTo( 42, 44);    // Top of lower control apron
    shape.lineTo(-22, 202);   // Top of slanted main face
    shape.lineTo(-85, 202);   // Back top roof
    shape.closePath();

    const bodyGeo = new THREE.ExtrudeGeometry(shape, { depth: 240, bevelEnabled: true, bevelThickness: 2, bevelSize: 1.5, bevelSegments: 3 });
    bodyGeo.center();
    // After Extrude along Z, rotate so extrusion is along X:
    const bodyMesh = new THREE.Mesh(bodyGeo, matCarbon);
    bodyMesh.rotation.y = -Math.PI / 2;
    bodyMesh.position.set(0, 101, -2.5);
    bodyMesh.castShadow = true;
    bodyMesh.receiveShadow = true;
    root.add(bodyMesh);

    // Helper for placing elements on the Upper Slanted Faceplate
    // Faceplate goes from (Z = +42, Y = 44) to (Z = -22, Y = 202)
    const upperAngle = Math.atan2(42 - (-22), 202 - 44); // ~22.05 deg
    function placeOnUpperFace(mesh, x, fracUp, zOffset) {
      const z0 = 42, y0 = 44;
      const z1 = -22, y1 = 202;
      const bz = z0 + (z1 - z0) * fracUp;
      const by = y0 + (y1 - y0) * fracUp;
      const nz = Math.cos(upperAngle), ny = Math.sin(upperAngle);
      mesh.position.set(x, by + ny * zOffset, bz + nz * zOffset);
      mesh.rotation.x = -upperAngle;
      root.add(mesh);
    }

    // Header Title Bar ([ GVK NEURAL ARCHIVE // UNIT 01 ]   [REV 2.6])
    const headerMat = new THREE.MeshBasicMaterial({ map: makeHeaderTexture() });
    const headerPlane = new THREE.Mesh(new THREE.PlaneGeometry(216, 18), headerMat);
    placeOnUpperFace(headerPlane, 0, 0.93, 2.2);

    // 2. LEFT VERTICAL SLIM BLU-RAY DRIVE
    const driveBezel = new THREE.Mesh(new THREE.BoxGeometry(62, 140, 6), matDarkTrim);
    placeOnUpperFace(driveBezel, -62, 0.48, 3.0);

    const driveSlot = new THREE.Mesh(new THREE.BoxGeometry(44, 126, 5), matCarbon);
    placeOnUpperFace(driveSlot, -62, 0.48, 5.5);

    const trayLip = new THREE.Mesh(new THREE.BoxGeometry(16, 114, 4), matDarkTrim);
    placeOnUpperFace(trayLip, -65, 0.48, 7.5);

    const ejectBtn = new THREE.Mesh(new THREE.BoxGeometry(7, 15, 4), matDarkTrim);
    placeOnUpperFace(ejectBtn, -48, 0.65, 7.5);

    const driveLed = new THREE.Mesh(new THREE.SphereGeometry(1.5, 12, 12), matGreenLED);
    placeOnUpperFace(driveLed, -48, 0.54, 8.0);

    // 3. RIGHT CARTRIDGE POCKET & PROTRUDING SIDE-BY-SIDE CASES
    const pocketFrame = new THREE.Mesh(new THREE.BoxGeometry(118, 104, 8), matDarkTrim);
    placeOnUpperFace(pocketFrame, 46, 0.56, 2.5);

    const pocketInterior = new THREE.Mesh(
      new THREE.BoxGeometry(108, 96, 10),
      new THREE.MeshBasicMaterial({ color: 0x050608 })
    );
    placeOnUpperFace(pocketInterior, 46, 0.56, 3.2);

    const cartridges = [];
    const specs = [
      { x: 12, num: '01', color: '#06b6d4', label: 'FRIEND 01' },
      { x: 46, num: '02', color: '#eab308', label: 'FRIEND 02' },
      { x: 80, num: '03', color: '#a855f7', label: 'FRIEND 03' }
    ];

    specs.forEach((s, idx) => {
      const spineTex = makeSpineTexture(s.num, s.color, s.label);
      const materials = [
        matDarkTrim, // right
        matDarkTrim, // left
        matDarkTrim, // top
        matDarkTrim, // bottom
        new THREE.MeshStandardMaterial({ map: spineTex, roughness: 0.4 }), // front spine
        matDarkTrim  // back
      ];
      const box = new THREE.Mesh(new THREE.BoxGeometry(29, 106, 62), materials);
      box.castShadow = true;
      placeOnUpperFace(box, s.x, 0.61, 28.0);
      cartridges.push(box);
    });

    // 4. STATUS LED BEZEL BELOW CARTRIDGES (SUBJ-01 / SUBJ-02 / SUBJ-03 / VAULT)
    const ledPanel = new THREE.Mesh(new THREE.BoxGeometry(92, 22, 5), matDarkTrim);
    placeOnUpperFace(ledPanel, 46, 0.13, 3.5);

    const statusLeds = [];
    for (let i = 0; i < 4; i++) {
      const isGreen = i < 2;
      const led = new THREE.Mesh(
        new THREE.SphereGeometry(2.8, 16, 16),
        isGreen ? matGreenLED : matRedLED
      );
      placeOnUpperFace(led, 15 + i * 21, 0.14, 6.2);
      statusLeds.push(led);
    }

    // 5. LOWER SLOPED CONTROL APRON
    // Goes from (Z = 90, Y = 16) to (Z = 42, Y = 44)
    const apronAngle = Math.atan2(90 - 42, 44 - 16); // ~59.75 deg from vertical
    function placeOnApron(mesh, x, fracUp, zOffset) {
      const z0 = 90, y0 = 16;
      const z1 = 42, y1 = 44;
      const bz = z0 + (z1 - z0) * fracUp;
      const by = y0 + (y1 - y0) * fracUp;
      const nz = Math.cos(apronAngle), ny = Math.sin(apronAngle);
      mesh.position.set(x, by + ny * zOffset, bz + nz * zOffset);
      mesh.rotation.x = -apronAngle;
      root.add(mesh);
    }

    // Online Green Pill LED
    const onlineLed = new THREE.Mesh(new THREE.BoxGeometry(5, 11, 3), matGreenLED);
    placeOnApron(onlineLed, -82, 0.48, 2.0);

    const onlineLight = new THREE.PointLight(0x22c55e, 0.8, 60);
    placeOnApron(onlineLight, -82, 0.48, 10.0);

    // Brushed Aluminum Station ID Plate
    const plateMat = new THREE.MeshStandardMaterial({
      map: makeSilverPlateTexture(),
      roughness: 0.3,
      metalness: 0.8
    });
    const idPlate = new THREE.Mesh(new THREE.BoxGeometry(66, 22, 2.5), plateMat);
    placeOnApron(idPlate, -8, 0.48, 2.0);

    // Kill-Switch Assembly (Right side of apron)
    const switchHex = new THREE.Mesh(new THREE.CylinderGeometry(8, 8, 5, 6), matChrome);
    placeOnApron(switchHex, 76, 0.42, 2.5);
    switchHex.rotation.x = Math.PI / 2 - apronAngle;

    const leverPivot = new THREE.Group();
    placeOnApron(leverPivot, 76, 0.42, 4.0);
    const leverMesh = new THREE.Mesh(new THREE.CylinderGeometry(2.2, 3, 18, 16), matChrome);
    leverMesh.position.set(0, 0, 9);
    leverMesh.rotation.x = Math.PI / 2 - 0.25;
    leverPivot.add(leverMesh);

    const redCoverPivot = new THREE.Group();
    placeOnApron(redCoverPivot, 76, 0.78, 4.0);
    const redCoverMesh = new THREE.Mesh(new THREE.BoxGeometry(20, 28, 14), matRedCover);
    redCoverMesh.position.set(0, 12, 6);
    redCoverPivot.add(redCoverMesh);

    // Interactive Controls
    let cart1Out = false;
    document.getElementById('btn-eject').addEventListener('click', () => {
      cart1Out = !cart1Out;
      const offset = cart1Out ? 65.0 : 28.0;
      const z0 = 42, y0 = 44, z1 = -22, y1 = 202, fracUp = 0.61;
      const bz = z0 + (z1 - z0) * fracUp;
      const by = y0 + (y1 - y0) * fracUp;
      const nz = Math.cos(upperAngle), ny = Math.sin(upperAngle);
      cartridges[0].position.set(12, by + ny * offset, bz + nz * offset);
    });

    let killActive = false;
    document.getElementById('btn-kill').addEventListener('click', () => {
      killActive = !killActive;
      leverMesh.rotation.x = Math.PI / 2 + (killActive ? 0.35 : -0.25);
      redCoverPivot.rotation.x = -apronAngle + (killActive ? -1.1 : 0.0);
      onlineLed.material = killActive ? matRedLED : matGreenLED;
      onlineLight.color.setHex(killActive ? 0xef4444 : 0x22c55e);
      statusLeds[0].material = killActive ? matRedLED : matGreenLED;
      statusLeds[1].material = killActive ? matRedLED : matGreenLED;
    });

    let wireframe = false;
    document.getElementById('btn-wire').addEventListener('click', () => {
      wireframe = !wireframe;
      matCarbon.wireframe = wireframe;
      matDarkTrim.wireframe = wireframe;
    });

    document.getElementById('btn-cam').addEventListener('click', () => {
      camera.position.set(-220, 215, 340);
      controls.target.set(0, 95, 0);
    });

    function animate() {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }
    animate();

    window.addEventListener('resize', () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    });
  </script>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[OK] Exported interactive Three.js HTML viewer: {output_path}")


if __name__ == "__main__":
    hw_dir = os.path.dirname(os.path.abspath(__file__))
    generate_all(hw_dir)
    generate_photoreal_viewer_html(os.path.join(hw_dir, "gvk_dock_viewer.html"))
