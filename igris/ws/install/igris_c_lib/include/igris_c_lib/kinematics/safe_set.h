#pragma once

#include <Eigen/Dense>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

class ISafeSet {
  public:
    virtual ~ISafeSet()                                           = default;
    virtual int Dimension() const                                 = 0;
    virtual bool IsInside(const Eigen::VectorXd &q) const         = 0;
    virtual double SignedDistance(const Eigen::VectorXd &q) const = 0;
};

class SafeSet2D : public ISafeSet {
  public:
    // Projection policy:
    // - kShortestDistance: nearest-point projection of desired_q to boundary.
    enum class ProjectionMode {
        kShortestDistance = 0,
        Intersection      = 1,
    };

    struct ProjectionResult {
        Eigen::Vector2d projection;  // Final projected point on boundary (or desired_q when already safe).
        int edge_index;              // Boundary edge index used for projection, -1 if not tied to an edge.
        bool is_valid;

        // True when current_q is outside safe set. In this case result is nearest projection of current_q.
        bool current_outside_error;

        // Distances to projected point, useful for PD control diagnostics.
        double current_to_projection_distance;
        double desired_to_projection_distance;

        ProjectionMode mode;
    };

    SafeSet2D() = default;
    explicit SafeSet2D(std::vector<Eigen::Vector2d> verts);

    bool IsValid() const;
    size_t Size() const;

    int Dimension() const override { return 2; }
    bool IsInside(const Eigen::VectorXd &q) const override;
    double SignedDistance(const Eigen::VectorXd &q) const override;

    bool IsInside(const Eigen::Vector2d &q) const;
    double SignedDistance(const Eigen::Vector2d &q) const;

    // Legacy API: project single point to nearest boundary.
    ProjectionResult ProjectToBoundary(const Eigen::Vector2d &q) const;

    // Extended API for control:
    // - Inputs: current_q and desired_q.
    // - Projection is nearest-point projection of desired_q to boundary.
    // - mode is kept for backward compatibility.
    ProjectionResult ProjectToBoundary(const Eigen::Vector2d &current_q, const Eigen::Vector2d &desired_q,
                                       ProjectionMode mode = ProjectionMode::kShortestDistance) const;

    ProjectionResult ProjectOutsideToBoundary(const Eigen::Vector2d &q) const;

    ProjectionResult ProjectToIntersection(const Eigen::Vector2d &current_q, const Eigen::Vector2d &desired_q) const;
    double GetMaxDistance() const;

    const std::vector<Eigen::Vector2d> &Vertices() const { return vertices_; }
    void SetVertices(std::vector<Eigen::Vector2d> verts);
    void UseSoftBoundary(double ratio = 0.95);

    static bool LoadFromJson(const std::string &path, SafeSet2D *out, std::string *error, const std::string &key = "vertices");

  private:
    struct SegmentProjection {
        Eigen::Vector2d projection;
        double distance;
    };

    static constexpr double kEps = 1e-12;

    static double Cross2(const Eigen::Vector2d &a, const Eigen::Vector2d &b);
    static SegmentProjection PointSegmentDistance(const Eigen::Vector2d &point, const Eigen::Vector2d &a, const Eigen::Vector2d &b);
    static bool InsideEvenOddStrict(const std::vector<Eigen::Vector2d> &verts, const Eigen::Vector2d &q);
    static ProjectionResult MakeProjectionResult(const Eigen::Vector2d &anchor, ProjectionMode mode);

    std::vector<Eigen::Vector2d> vertices_;
    double max_distance_ = 0.0;
};

class SafeSetManager {
  public:
    bool LoadPolygon2DFromJson(const std::string &name, const std::string &path, std::string *error, const std::string &key = "vertices",
                               double soft_boundary_ratio = 1.0);

    const ISafeSet *Get(const std::string &name) const;
    template <typename T> const T *GetAs(const std::string &name) const;
    bool Has(const std::string &name) const;
    size_t Size() const;
    void Clear();

  private:
    std::unordered_map<std::string, std::unique_ptr<ISafeSet>> sets_;
};

template <typename T> const T *SafeSetManager::GetAs(const std::string &name) const {
    auto it = sets_.find(name);
    if (it == sets_.end())
        return nullptr;
    return dynamic_cast<const T *>(it->second.get());
}
