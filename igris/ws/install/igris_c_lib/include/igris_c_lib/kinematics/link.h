#ifndef __LINK_C54B4148_B9FA_428C_BB09_9D8D7489BFAC_H__
#define __LINK_C54B4148_B9FA_428C_BB09_9D8D7489BFAC_H__

#include "igris_c_lib/common/math_type_define.h"
#include "igris_c_shm/logger/logger.h"

#include <Eigen/Dense>
#include <array>
#include <limits>
#include <rbdl/addons/urdfreader/urdfreader.h>
#include <rbdl/rbdl.h>
#include <vector>

enum { JOINT_FLOATING_BASE, JOINT_6DOF, JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_FIXED };

class Joint {
  public:
    Joint();
    Joint(int joint_type);
    Joint(int joint_type, const Eigen::Vector3d &joint_axis);
    int joint_type_;
    Eigen::Vector3d joint_axis_;
    Eigen::Vector3d joint_translation_;
    Eigen::Matrix3d joint_rotation_;
    Eigen::Vector3d parent_translation_;
    Eigen::Matrix3d parent_rotation_;
    int joint_id_;
    // RigidBodyDynamics::Joint ToRBDLJoint();
};

class Link {
  public:
    Link();
    Link(RigidBodyDynamics::Model &model_, const unsigned int link_id);
    ~Link();

    std::string name_;

    Eigen::MatrixXd jac_;
    Eigen::MatrixXd jac_pre_;
    Eigen::MatrixXd jac_dot_;

    Eigen::MatrixXd jac_com_;

    Eigen::Vector3d xpos;
    Eigen::Vector3d xipos;
    Eigen::Vector3d v;
    Eigen::Vector3d vi;
    Eigen::Vector3d w;
    Eigen::Matrix3d rotm;
    Eigen::Vector3d rpy;

    Eigen::Vector3d joint_trans;
    Eigen::Matrix3d joint_rotm;

    Eigen::Vector3d parent_trans;
    Eigen::Matrix3d parent_rotm;

    RigidBodyDynamics::Math::SpatialVector vw;

    Eigen::Vector3d com_position_l_;
    Eigen::Matrix3d inertia;

    Eigen::Vector3d link_pos_offset_;  // for hand/foot offset

    // id of body from rbdl
    int body_id_;

    // id of parent link
    int parent_id_;

    std::vector<int> child_id_;

    // id of current link
    int link_id_;
    int link_id_original_;

    double mass;

    void UpdateAll(RigidBodyDynamics::Model &model_, const Eigen::VectorXd &q_virtual_, const Eigen::VectorXd &q_dot_virtual_);

    /*
    Update Position and Velocity data of link
    xpos, rotm, v, w
    */
    void UpdatePos(RigidBodyDynamics::Model &model_, const Eigen::VectorXd &q_virtual_, const Eigen::VectorXd &q_dot_virtual_);

    /*
    Upate Jacobian and COM jacobian of link
    */
    void UpdateJac(RigidBodyDynamics::Model &model_, const Eigen::VectorXd &q_virtual_);

    /*
    Get Jacobian of Specific position
    */
    Eigen::MatrixXd GetPointJac(RigidBodyDynamics::Model &model_, const Eigen::VectorXd &q_virtual_, const Eigen::Vector3d &point);

    // /*
    // Get Position of
    // */
    void GetPointPos(RigidBodyDynamics::Model &model_, const Eigen::VectorXd &q_virtual_, const Eigen::VectorXd &q_dot_virtual_,
                     Eigen::Vector3d &local_pos, Eigen::Vector3d &global_pos);

    Eigen::Quaterniond GetQuat();

    // void UpdateJacDot(RigidBodyDynamics::Model &model_, const Eigen::VectorXd
    // &q_virtual, const Eigen::VectorXd &q_dot_virtual);
    Eigen::MatrixXd GetJacDot(RigidBodyDynamics::Model &model_, const Eigen::VectorXd &q_virtual, const Eigen::VectorXd &q_dot_virtual);

    // Get spatial transform matrix 6x6 from link to base, Matrix *
    // local_spacial_vector = global_spacial_vector
    Eigen::MatrixXd GetSpatialTranform();

    // Get adjoint matrix 6x6 from link to base, Matrix * local_spacial_vector =
    // global_spatial_vector
    Eigen::MatrixXd GetAdjointMatrix();

    // Get Inertia Matrix of link from global frame
    Eigen::MatrixXd GetSpatialInertiaMatrix(bool rotation_first = true);

    RigidBodyDynamics::Body ToRBDLBody();

    // Combine two link. Add link_add to current link. rotm and xpos is the
    // position of link_add in current link frame
    void AddLink(Link &link_add, Eigen::Matrix3d &rotm, Eigen::Vector3d &xpos);

    // Print Link Data
    void Print();

    double cutoff_freq_;
    double damping_;
    double sampling_frequency_;

    std::vector<Eigen::Vector3d> vp_;
    std::vector<Eigen::Vector3d> vlpf_;

    // velocity lpf
    void initialize2lpf(double cutoff_freq, double damping, double dt);
    void calc2lpf();
};

class Parallel4BarLink {
  private:
    RigidBodyDynamics::Model *model_;
    std::string name_;
    robros::shm::logger::Logger *logger_ = nullptr;
    unsigned int p_link_2_id_            = 2;
    bool roll_pitch_intersecting_        = true;

    Eigen::Vector2d q_last_     = Eigen::Vector2d::Zero();
    Eigen::Vector2d theta_last_ = Eigen::Vector2d::Zero();
    double lm_lambda_           = 1e-4;  // initial LM damping
    Eigen::Vector2d dq_last_    = Eigen::Vector2d::Zero();
    Eigen::Vector2d motor_last_ = Eigen::Vector2d::Zero();
    bool has_prev_              = false;

    static inline double wrapToPi(double a);
    static inline Eigen::Vector2d wrapToPiVec(const Eigen::Vector2d &v);
    static inline Eigen::Vector2d clampToLimits(const Eigen::Vector2d &q, const Eigen::Vector2d &qmin, const Eigen::Vector2d &qmax);
    static bool IsWristLinkName(const std::string &name);
    static bool IsWaistLinkName(const std::string &name);

    enum class FkLinkGroup { Waist, Ankle, Wrist, Other };

    struct FkGroupTuning {
        double armijo_c_search               = 1e-4;
        double armijo_beta_search            = 0.5;
        double armijo_c_refine               = 1e-3;
        double armijo_beta_refine            = 0.6;
        double max_step                      = 0.1;
        double fallback_step                 = 0.01;
        int max_ls                           = 36;
        int max_ls_fast                      = 8;
        int refine_max_itr                   = 8;
        int refine_max_ls                    = 4;
        double velocity_dq_abs_tol           = 1e-3;
        double velocity_dq_rel_tol           = 1.0;
        double velocity_direction_cos_min    = 0.0;
        double velocity_max_step_ratio       = 2.0;
        double velocity_min_step_ratio       = 0.25;
        double velocity_step_abs_margin      = 2e-3;
        double rescue_sigma_parallel_scale   = 0.5;
        double rescue_sigma_orthogonal_scale = 0.15;
        int rescue_sample_count              = 8;
    };

    struct FkContext {
        Eigen::Vector2d motor_angles = Eigen::Vector2d::Zero();
        FkLinkGroup group            = FkLinkGroup::Other;
        FkGroupTuning tuning;

        bool is_wrist_group = false;
        bool is_ankle_group = false;
        bool is_waist_group = false;
        bool first_tick     = true;

        int max_itr                        = 50;
        int emergency_switch_iters         = 3;
        double alpha_min                   = 1e-10;
        double min_q_progress              = 2e-4;
        double motor_move_eps              = 1e-4;
        double continuity_w                = 0.0;
        double continuity_theta_w          = 0.0;
        double singular_cond_thres         = 40.0;
        double singular_lambda_mul         = 8.0;
        double tol_r                       = 1e-4;
        double residual_soft_limit         = 5e-3;
        double residual_target_limit       = 1e-4;
        double residual_hard_limit         = 1e-4;
        double residual_degraded_limit     = 5e-3;
        double tol_r_loose                 = 3e-4;
        double tol_r_restart_stop          = 3e-4;
        double gdot_skip_thresh            = 5e-10;
        int nonmono_window                 = 4;
        double nonmono_margin              = 1e-8;
        double fast_max_motor_jump         = 0.05;
        double fast_max_continuous_q_step  = 0.08;
        double control_dt                  = 1.0 / 300.0;
        double velocity_dq_eps             = 1e-6;
        double velocity_direction_eps      = 1e-5;
        double velocity_guard_cond_max     = 80.0;
        double velocity_jacobian_damping   = 1e-8;
        double score_residual_w            = 1.0;
        double score_velocity_w            = 2.0;
        double score_bad_direction_penalty = 1.0;
        double score_step_ratio_penalty_w  = 0.5;

        double armijo_c_search    = 1e-4;
        double armijo_beta_search = 0.5;
        double armijo_c_refine    = 1e-3;
        double armijo_beta_refine = 0.6;
        double max_step           = 0.1;
        double fallback_step      = 0.01;
        int max_ls                = 36;
        int max_ls_fast           = 8;
        int refine_max_itr        = 8;
        int refine_max_ls         = 4;

        Eigen::Vector2d velocity_guard_dq_expected = Eigen::Vector2d::Zero();
        bool velocity_guard_active                 = false;
    };

    struct FkCandidate {
        Eigen::Vector2d q     = Eigen::Vector2d::Zero();
        Eigen::Vector2d theta = Eigen::Vector2d::Zero();
        double lambda         = 1e-4;
        double score          = std::numeric_limits<double>::infinity();
        double residual_norm  = std::numeric_limits<double>::infinity();
        int residual_bucket   = 2;
    };

    struct SeedSolveResult {
        Eigen::Vector2d q          = Eigen::Vector2d::Zero();
        Eigen::Vector2d theta      = Eigen::Vector2d::Zero();
        double lambda              = 1e-4;
        double alpha_last_accepted = 1.0;
        bool solved_restart        = false;
    };

    struct FkSavedState {
        Eigen::Vector2d q;
        Eigen::Vector2d theta;
        Eigen::Vector2d dq;
        Eigen::Vector2d motor;
        bool has_prev;
        double lambda;
    };

    void clampFkMotorAngles(Eigen::Vector2d &motor_angles) const;
    FkContext makeFkContext(const Eigen::Vector2d &motor_angles);
    double fkResidualNorm(const FkContext &ctx, const Eigen::Vector2d &theta_candidate) const;
    bool fkResidualOk(const FkContext &ctx, const Eigen::Vector2d &theta_candidate, double residual_limit) const;
    bool persistFkCandidate(const FkContext &ctx, const Eigen::Vector2d &q_solution, double lambda_solution, double residual_limit);
    void refineFkCandidate(const FkContext &ctx, Eigen::Vector2d &q, Eigen::Vector2d &theta, double &lambda, double alpha_seed);
    bool polishFkCandidate(const FkContext &ctx, Eigen::Vector2d &q, Eigen::Vector2d &theta, double &lambda);
    Eigen::Vector2d clampToVelocityTrust(const FkContext &ctx, const Eigen::Vector2d &q_candidate, const Eigen::Vector2d &q_reference,
                                         double trust_radius) const;
    double fkCandidateScore(const FkContext &ctx, const Eigen::Vector2d &q_candidate, const Eigen::Vector2d &theta_candidate,
                            const Eigen::Vector2d &q_reference) const;
    bool fkVelocityOk(const FkContext &ctx, const Eigen::Vector2d &q_candidate, const Eigen::Vector2d &q_reference) const;
    std::vector<Eigen::Vector2d> buildFkSeeds(const FkContext &ctx);
    void updateFkBestCandidate(const FkContext &ctx, FkCandidate &best, const Eigen::Vector2d &q_candidate,
                               const Eigen::Vector2d &theta_candidate, double lambda_candidate) const;
    SeedSolveResult solveFromFkSeed(const FkContext &ctx, const Eigen::Vector2d &seed);
    bool tryLocalContinuationPersist(const FkContext &ctx);
    bool tryPredictedRescuePersist(const FkContext &ctx);
    bool finalizeFkCandidate(const FkContext &ctx, const FkCandidate &best);

  public:
    // Assumption //
    // 1: motor rotates in +y axis
    // 2: All the local frames are aligned in orientation at zero pose
    // 3: motor rotates in [-90 90]degree
    Parallel4BarLink();
    Parallel4BarLink(RigidBodyDynamics::Model *model, const std::string &name, const Eigen::Matrix3x2d &r_a_i,
                     const Eigen::Matrix3x2d &r_b_i, const Eigen::Matrix3x2d &r_c_i);
    ~Parallel4BarLink() = default;

    Eigen::Vector2d ik(Eigen::Vector2d joint_angles_in);
    Eigen::Matrix2d jac(Eigen::Vector2d joint_angles,
                        Eigen::Vector2d motor_angles);  // J_c: from joint vel to motor vel
    Eigen::Vector2d fk_legacy(Eigen::Vector2d motor_angles);
    Eigen::Vector2d fk(Eigen::Vector2d motor_angles);

    void setLogger(robros::shm::logger::Logger &l) { logger_ = &l; }
    robros::shm::logger::Logger *getLogger() const { return logger_; }

    bool verbose_     = true;
    bool is_elbow_up_ = true;  // Is the bar.cross(rod)(1) aligned with the motor axis (y-axis)?
    Eigen::Vector2d motor_position_, joint_position_;
    Eigen::Vector2d motor_velocity_, joint_velocity_;
    Eigen::Vector2d motor_torque_, joint_torque_;

    // values which have to be initialized at constructor
    Eigen::Matrix3x2d r_a_init_, r_b_init_, r_c_init_;

    // calculate in the constructor
    Eigen::Vector2d l_bar_, l_rod_;
    Eigen::Matrix3x2d r_c_offset_local_;
    Eigen::Matrix3x2d r_b_offset_local_;

    Eigen::Vector2d motor_angles_min_, motor_angles_max_, joint_angles_min_, joint_angles_max_;

    // ik
    Eigen::Matrix3x2d a_vec_, b_vec_;
    Eigen::Matrix3x2d r_c_;
    Eigen::Vector3d xpos_last_;
    Eigen::Matrix3d rotm_last_;

    // jac
    Eigen::Matrix3x2d r_b_;
    Eigen::Matrix3x2d r_bar_, r_rod_;

    Eigen::Matrix2x6d J_x_;
    Eigen::Matrix2d J_theta_;
    Eigen::Matrix2x6d J_;
    Eigen::Matrix2d J_c_;
};

enum class ParallelLinkCommandMode { MOTOR_SPACE, JOINT_SPACE };
enum class ParallelLinkIKSolverMode { GLOBAL, LOCAL };
enum class ParallelLinkSafetyMode { CLIP, SATURATE };
enum class ParallelLinkSoftSafeSetMode : bool { SOFT = true, HARD = false };
enum class ParallelLinkModeIndex : size_t {
    WAIST       = 0,
    LEFT_ANKLE  = 1,
    RIGHT_ANKLE = 2,
    LEFT_WRIST  = 3,
    RIGHT_WRIST = 4,
};

static constexpr size_t kParallelLinkModeCount                                               = 5;
static constexpr double kPlaHardSafeSetCoef                                                  = 0.95;
static constexpr double kPlaSoftSafeSetCoef                                                  = 0.8;
static constexpr std::array<ParallelLinkModeIndex, kParallelLinkModeCount> kAllParallelModes = {
    ParallelLinkModeIndex::WAIST,      ParallelLinkModeIndex::LEFT_ANKLE,  ParallelLinkModeIndex::RIGHT_ANKLE,
    ParallelLinkModeIndex::LEFT_WRIST, ParallelLinkModeIndex::RIGHT_WRIST,
};

enum class ParallelKinematicMode { MOTOR_SPACE, JOINT_SPACE };
enum class ParallelIKSolverMode { GLOBAL, LOCAL };
enum class ParallelSafetyMode { CLIP, SATURATE };
enum class ParallelSoftSafeSetMode : bool { SOFT = true, HARD = false };
enum class ParallelLinkTarget { WAIST = 0, LEFT_ANKLE, RIGHT_ANKLE, LEFT_WRIST, RIGHT_WRIST };

static constexpr std::array<ParallelLinkTarget, kParallelLinkModeCount> kAllParallelTargets = {
    ParallelLinkTarget::WAIST,      ParallelLinkTarget::LEFT_ANKLE,  ParallelLinkTarget::RIGHT_ANKLE,
    ParallelLinkTarget::LEFT_WRIST, ParallelLinkTarget::RIGHT_WRIST,
};

struct ParallelSpec {
    const char *name;
    RigidBodyDynamics::Model *model;
    ParallelLinkModeIndex mode_index;
    bool is_elbow_up;
    Eigen::Vector2d motor_min;
    Eigen::Vector2d motor_max;
    Eigen::Vector2d joint_min;
    Eigen::Vector2d joint_max;
    double a0_x, a0_y, a0_z;
    double b0_x, b0_y, b0_z;
    double c0_x, c0_y, c0_z;
    double a1_x, a1_y, a1_z;
    double b1_x, b1_y, b1_z;
    double c1_x, c1_y, c1_z;

    void Apply(std::array<Parallel4BarLink, kParallelLinkModeCount> &links, Eigen::Matrix3x2d &r_a_i, Eigen::Matrix3x2d &r_b_i,
               Eigen::Matrix3x2d &r_c_i) const {
        r_a_i.col(0) << a0_x, a0_y, a0_z;
        r_b_i.col(0) << b0_x, b0_y, b0_z;
        r_c_i.col(0) << c0_x, c0_y, c0_z;

        r_a_i.col(1) << a1_x, a1_y, a1_z;
        r_b_i.col(1) << b1_x, b1_y, b1_z;
        r_c_i.col(1) << c1_x, c1_y, c1_z;

        Parallel4BarLink &target_link = links[static_cast<size_t>(mode_index)];
        target_link                   = Parallel4BarLink(model, name, r_a_i, r_b_i, r_c_i);
        target_link.is_elbow_up_      = is_elbow_up;
        target_link.verbose_          = false;
        target_link.motor_angles_min_ << motor_min(0), motor_min(1);
        target_link.motor_angles_max_ << motor_max(0), motor_max(1);
        target_link.joint_angles_min_ << joint_min(0), joint_min(1);
        target_link.joint_angles_max_ << joint_max(0), joint_max(1);
    }
};

#endif /* __LINK_C54B4148_B9FA_428C_BB09_9D8D7489BFAC_H__ */
