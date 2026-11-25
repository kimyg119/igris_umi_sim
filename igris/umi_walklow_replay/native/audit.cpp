// Vendor-parity audit for the UMI replay scene, run on the vendor libmujoco 3.2.7.
//
// Robot checks are the same as igris/isaac_vendor_room/observer.cpp: every robot
// mass, inertia, frame, joint, actuator, equality, collision geometry, contact
// parameter, keyframe and solver option must equal the reference model; only
// the sparse Jacobian and the hand/environment contact bits may differ.
// The scene checks are general: any number of free "room_" bodies, each made of
// box colliders that robot and hands can touch.
#include <mujoco/mujoco.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void require(bool ok, const char* field) {
    if (!ok) { std::fprintf(stderr, "Vendor parity failed: %s\n", field); std::exit(1); }
}

int main(int argc, char** argv) {
    if (argc != 4) { std::fprintf(stderr, "usage: audit reference.xml scene.xml model.json\n"); return 2; }
    require(std::strcmp(mj_versionString(), "3.2.7") == 0, "MuJoCo version");
    char error[4096]{};
    auto a = mj_loadXML(argv[1], nullptr, error, sizeof(error));
    if (!a) { std::fprintf(stderr, "%s\n", error); return 1; }
    auto b = mj_loadXML(argv[2], nullptr, error, sizeof(error));
    if (!b) { std::fprintf(stderr, "%s\n", error); return 1; }
    require(b->nu == a->nu && b->neq == a->neq, "actuator/equality counts");
    int original_jacobian = a->opt.jacobian;
    require(b->opt.jacobian == a->opt.jacobian || b->opt.jacobian == mjJAC_SPARSE, "authorized dense/sparse representation only");
    a->opt.jacobian = b->opt.jacobian;
    require(std::memcmp(&a->opt, &b->opt, sizeof(mjOption)) == 0, "manufacturer solver options except the Jacobian representation");
    #define SAME(field, n) require(std::memcmp(a->field, b->field, sizeof(*a->field) * (n)) == 0, #field)
    SAME(body_mass, a->nbody); SAME(body_inertia, a->nbody * 3);
    SAME(body_ipos, a->nbody * 3); SAME(body_iquat, a->nbody * 4);
    SAME(body_pos, a->nbody * 3); SAME(body_quat, a->nbody * 4);
    SAME(jnt_type, a->njnt); SAME(jnt_pos, a->njnt * 3); SAME(jnt_axis, a->njnt * 3); SAME(jnt_range, a->njnt * 2);
    SAME(jnt_stiffness, a->njnt); SAME(qpos_spring, a->nq);
    SAME(dof_armature, a->nv); SAME(dof_damping, a->nv); SAME(dof_frictionloss, a->nv);
    SAME(actuator_gear, a->nu * 6); SAME(actuator_gainprm, a->nu * mjNGAIN); SAME(actuator_biasprm, a->nu * mjNBIAS);
    SAME(actuator_ctrlrange, a->nu * 2); SAME(actuator_forcerange, a->nu * 2);
    SAME(eq_type, a->neq); SAME(eq_data, a->neq * mjNEQDATA); SAME(eq_solref, a->neq * mjNREF); SAME(eq_solimp, a->neq * mjNIMP);
    SAME(qpos0, a->nq);
    for (int g = 0; g < a->ngeom; g++) {
        int body = a->geom_bodyid[g];
        int h = b->body_geomadr[body] + g - a->body_geomadr[body];
        #define GEOM(field, n) require(std::memcmp(a->field + (g) * (n), b->field + (h) * (n), sizeof(*a->field) * (n)) == 0, #field)
        GEOM(geom_type, 1); GEOM(geom_size, 3); GEOM(geom_pos, 3); GEOM(geom_quat, 4);
        GEOM(geom_friction, 3); GEOM(geom_solref, mjNREF); GEOM(geom_solimp, mjNIMP); GEOM(geom_condim, 1);
        bool hand = false;
        for (int ancestor = body; ancestor; ancestor = a->body_parentid[ancestor]) {
            auto name = mj_id2name(a, mjOBJ_BODY, ancestor);
            if (name && (!std::strcmp(name, "l_hand_rcv") || !std::strcmp(name, "r_hand_rcv"))) hand = true;
        }
        // Hand geoms never collide in the reference.  In the scene they either stay
        // that way (meshes) or take the environment-only bit (contact pads).
        if (hand) require(a->geom_contype[g] == 0 && a->geom_conaffinity[g] == 0 && b->geom_conaffinity[h] == 0
                          && (b->geom_contype[h] == 0 || b->geom_contype[h] == 2), "authorized hand environment-only contact");
        else { GEOM(geom_contype, 1); GEOM(geom_conaffinity, 1); }
    }
    // Every original robot/ground collision-filter relationship is unchanged.
    for (int g = 0; g < a->ngeom; g++) for (int h = 0; h < a->ngeom; h++) {
        auto map = [&](int k) { int body = a->geom_bodyid[k]; return b->body_geomadr[body] + k - a->body_geomadr[body]; };
        int x = map(g), y = map(h);
        bool before = (a->geom_contype[g] & a->geom_conaffinity[h]) || (a->geom_contype[h] & a->geom_conaffinity[g]);
        bool after = (b->geom_contype[x] & b->geom_conaffinity[y]) || (b->geom_contype[y] & b->geom_conaffinity[x]);
        require(before == after, "original robot and ground collision eligibility");
    }
    for (int k = 0; k < a->nkey; k++)
        require(std::memcmp(a->key_qpos + k * a->nq, b->key_qpos + k * b->nq, sizeof(mjtNum) * a->nq) == 0, "robot keyframes");
    // Scene: only free, box-shaped room bodies were added after the robot.
    int room = b->nbody - a->nbody;
    require(room >= 1 && b->nq == a->nq + room * 7 && b->nv == a->nv + room * 6, "added bodies are free room bodies only");
    require(b->ngeom > a->ngeom, "room colliders present");
    for (int i = a->nbody; i < b->nbody; i++) {
        const char* name = mj_id2name(b, mjOBJ_BODY, i);
        require(name && !std::strncmp(name, "room_", 5), "room body naming");
        require(b->body_parentid[i] == 0 && b->body_jntnum[i] == 1 && b->jnt_type[b->body_jntadr[i]] == mjJNT_FREE, "room body is free");
        require(b->body_geomnum[i] >= 1 && b->body_mass[i] > 0, "room body has mass and colliders");
        for (int g = b->body_geomadr[i]; g < b->body_geomadr[i] + b->body_geomnum[i]; g++)
            require(b->geom_type[g] == mjGEOM_BOX && b->geom_contype[g] == 1 && b->geom_conaffinity[g] == 3,
                    "room colliders are boxes that robot and hands can touch");
    }
    for (int g = a->ngeom; g < b->ngeom; g++)
        require(b->geom_bodyid[g] >= a->nbody, "no static geometry was added to the world");
    auto d = mj_makeData(b); mj_forward(b, d);
    FILE* f = std::fopen(argv[3], "w"); if (!f) return 1;
    std::fprintf(f, "{\"vendor_robot_parameters_pass\":true,\"room_free_bodies\":%d,\"jacobian_changed\":%s,\"mujoco_version\":\"%s\",\"timestep\":%.17g,\"nq\":%d,\"nv\":%d,\"nu\":%d,\"neq\":%d,\"bodies\":[",
                 room, b->opt.jacobian == original_jacobian ? "false" : "true", mj_versionString(), b->opt.timestep, b->nq, b->nv, b->nu, b->neq);
    for (int i = 0; i < b->nbody; i++) {
        auto name = mj_id2name(b, mjOBJ_BODY, i);
        std::fprintf(f, "%s{\"name\":\"%s\",\"pose\":[", i ? "," : "", name ? name : "");
        for (int k = 0; k < 7; k++) std::fprintf(f, "%s%.17g", k ? "," : "", k < 3 ? d->xpos[3 * i + k] : d->xquat[4 * i + k - 3]);
        std::fprintf(f, "]}");
    }
    std::fprintf(f, "],\"hinge_q\":{"); bool comma = false;
    for (int j = 0; j < b->njnt; j++) if (b->jnt_type[j] == mjJNT_HINGE) {
        const char* name = mj_id2name(b, mjOBJ_JOINT, j); if (!name) continue;
        std::fprintf(f, "%s\"%s\":%.17g", comma ? "," : "", name, d->qpos[b->jnt_qposadr[j]]); comma = true;
    }
    std::fprintf(f, "}}\n"); std::fclose(f);
    std::printf("Vendor 3.2.7 robot parameters PASS: nq=%d nv=%d nu=%d neq=%d; %d free room bodies; Jacobian changed=%s\n",
                b->nq, b->nv, b->nu, b->neq, room, b->opt.jacobian == original_jacobian ? "false" : "true");
    mj_deleteData(d); mj_deleteModel(a); mj_deleteModel(b);
}
