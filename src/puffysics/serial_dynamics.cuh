#pragma once
#include "contact_traits.cuh"

// Dynamics for serial revolute chains, including fixed attachments. Callers
// supply world poses, joint origins/axes and model data; no scene or FK
// ownership. Model::last(body) is the final influencing joint (-1 for a fixed
// root). Model supplies mass(body), local com(body), and a full symmetric
// Tensor. Unit axes, finite physical data, and nonnegative armature are
// prerequisites. This explicit path omits velocity bias, external wrenches and
// implicit motors; the compiled native articulated solver remains available for
// those features.
template <int N, class T = PfContactTraits> struct PfSerialDynamicsT {
  static_assert(N > 0, "A chain needs at least one coordinate");
  using Vec3 = typename T::Vec3;
  struct Tensor {
    float xx, yy, zz, xy, xz, yz;
  };
  template <class Model, class Pose>
  __device__ static void mass(const Pose *bodies, const Vec3 *origins,
                              const Vec3 *axes, int body_count,
                              const float *armature, float matrix[N][N]) {
    for (int row = 0; row < N; ++row) {
      for (int column = 0; column < N; ++column) {
        matrix[row][column] = 0.0f;
      }
    }
    for (int body = 0; body < body_count; ++body) {
      float mass = Model::mass(body);
      Tensor inertia = Model::inertia(body);
      Vec3 com = T::add(bodies[body].position,
                        T::rotate(bodies[body].rotation, Model::com(body)));
      int last = Model::last(body);
      for (int row = 0; row <= last; ++row) {
        Vec3 linear_row = T::cross(axes[row], T::sub(com, origins[row]));
        Vec3 local_axis =
            T::rotate(T::conjugate(bodies[body].rotation), axes[row]);
        Vec3 local_inertia =
            T::v3(inertia.xx * local_axis.x + inertia.xy * local_axis.y +
                      inertia.xz * local_axis.z,
                  inertia.xy * local_axis.x + inertia.yy * local_axis.y +
                      inertia.yz * local_axis.z,
                  inertia.xz * local_axis.x + inertia.yz * local_axis.y +
                      inertia.zz * local_axis.z);
        Vec3 inertia_row = T::rotate(bodies[body].rotation, local_inertia);
        for (int column = 0; column <= row; ++column) {
          Vec3 linear_column =
              T::cross(axes[column], T::sub(com, origins[column]));
          float value = mass * T::dot(linear_row, linear_column) +
                        T::dot(axes[column], inertia_row);
          matrix[row][column] += value;
          if (row != column) {
            matrix[column][row] += value;
          }
        }
      }
    }
    for (int joint = 0; joint < N; ++joint) {
      matrix[joint][joint] += armature[joint];
    }
  }
  // Floor is caller policy, matching legacy regularized Cholesky when desired.
  // Only the lower triangle of the output is initialized/consumed.
  __device__ static void factor(const float matrix[N][N], float lower[N][N],
                                float diagonal_floor = 1.0e-8f) {
    for (int row = 0; row < N; ++row) {
      for (int column = 0; column <= row; ++column) {
        float sum = matrix[row][column];
        for (int k = 0; k < column; ++k) {
          sum -= lower[row][k] * lower[column][k];
        }
        if (row == column) {
          lower[row][column] = sqrtf(T::max(sum, diagonal_floor));
        } else {
          lower[row][column] = sum / lower[column][column];
        }
      }
    }
  }
  __device__ static void solve(const float lower[N][N], const float rhs[N],
                               float solution[N]) {
    float y[N] = {0};
    for (int row = 0; row < N; ++row) {
      float sum = rhs[row];
      for (int k = 0; k < row; ++k) {
        sum -= lower[row][k] * y[k];
      }
      y[row] = sum / lower[row][row];
    }
    for (int row = N - 1; row >= 0; --row) {
      float sum = y[row];
      for (int k = row + 1; k < N; ++k) {
        sum -= lower[k][row] * solution[k];
      }
      solution[row] = sum / lower[row][row];
    }
  }
  template <class Model, class Pose>
  __device__ static void gravity(const Pose *bodies, const Vec3 *origins,
                                 const Vec3 *axes, int body_count, Vec3 gravity,
                                 float torque[N]) {
    for (int joint = 0; joint < N; ++joint) {
      torque[joint] = 0.0f;
    }
    for (int body = 0; body < body_count; ++body) {
      float mass = Model::mass(body);
      Vec3 com = T::add(bodies[body].position,
                        T::rotate(bodies[body].rotation, Model::com(body)));
      int last = Model::last(body);
      for (int joint = 0; joint <= last; ++joint) {
        Vec3 linear = T::cross(axes[joint], T::sub(com, origins[joint]));
        torque[joint] += mass * T::dot(linear, gravity);
      }
    }
  }
  __device__ static void response(const float lower[N][N], const Vec3 *origins,
                                  const Vec3 *axes, int last_joint, Vec3 point,
                                  Vec3 linear_direction, Vec3 angular_direction,
                                  float jacobian[N], float response[N],
                                  float *inverse_mass) {
    for (int joint = 0; joint < N; ++joint) {
      if (joint <= last_joint) {
        Vec3 linear = T::cross(axes[joint], T::sub(point, origins[joint]));
        jacobian[joint] = T::dot(linear, linear_direction) +
                          T::dot(axes[joint], angular_direction);
      } else {
        jacobian[joint] = 0.0f;
      }
      response[joint] = 0.0f;
    }
    solve(lower, jacobian, response);
    *inverse_mass = 0.0f;
    for (int joint = 0; joint < N; ++joint) {
      *inverse_mass += jacobian[joint] * response[joint];
    }
  }
  __device__ static Vec3 point_velocity(const float *qd, const Vec3 *origins,
                                        const Vec3 *axes, int last_joint,
                                        Vec3 point) {
    Vec3 velocity = T::v3(0, 0, 0);
    for (int joint = 0; joint <= last_joint; ++joint) {
      velocity =
          T::add(velocity,
                 T::scale(T::cross(axes[joint], T::sub(point, origins[joint])),
                          qd[joint]));
    }
    return velocity;
  }
  __device__ static Vec3 angular_velocity(const float *qd, const Vec3 *axes,
                                          int last_joint) {
    Vec3 velocity = T::v3(0, 0, 0);
    for (int joint = 0; joint <= last_joint; ++joint) {
      velocity = T::add(velocity, T::scale(axes[joint], qd[joint]));
    }
    return velocity;
  }
};
