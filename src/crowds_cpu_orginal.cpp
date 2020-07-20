// Position-Based Real-Time Simulation of Large Crowds
// Copyright (c) 2020, Tomer Weiss
//
// Redistribution and use in source and binary forms, with or without modification, are
// permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this list of
//    conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list
//    of conditions and the following disclaimer in the documentation and/or other materials
//    provided with the distribution.
// THIS SOFTWARE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR  A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
// OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
// IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
// OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Original author: Tomer Weiss <http://www.cs.ucla.edu/~tweiss>
#include <stdio.h>
#include <string>
#include <time.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>
#include <tuple> //added this
#include <complex>

#include <stdio.h>
#include <GLFW/glfw3.h>

#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#include <GL/glut.h>
#endif
#include <sys/time.h>

#define OUT_PATH "../out/"

/* ========================= */
/* Simulation Engine Params: */
#define BLOCK_SIZE 256
#define _M_PI 3.14159265358979323846f
#define K_NOT_USED -1
#define _EPSILON 0.00001f
#define EPS 0.0000097
#define MS_PER_UPDATE 0.02 // 0.018
#define KSI 0.01 // 0.0093 //0.005/0.54

#define ALPHA 1.2
#define ITER_COUNT 1
#define MAX_ACCEL 20.0f
#define MAX_SPEED 10.4f
#define V_PREF_ACCEL 1.4f
#define KSI_ACCEL 0.54f
#define NN_ACCEL 10.0f
/* ========================= */
/* Scenario Params: */
#define WIDTH 1280
#define HEIGHT 720
#define ROWS 16
#define COLS 36
#define GROUND_HEIGHT 45.0
#define GRID_UP_BOUND -436.0
#define GRID_LOW_BOND GROUND_HEIGHT + 20
#define LEFT_BOUND_X -285.0
#define RIGHT_BOUND_X 285.0
/* ========================= */


typedef unsigned char BYTE;
typedef unsigned int uint;

//struct float2 { int x; int y; };
//typedef std::complex<float> float2;

#define __device static inline
struct float2 {
   float x, y;
   float operator[](int i) const { return *(&x + i); }
   float& operator[](int i) { return *(&x + i); }
};

__device float2 make_float2(float x, float y)
{
     float2 a = {x, y};
     return a;
}

struct float3 {
     float x, y, z;
 #ifdef WITH_OPENCL
     float w;
 #endif
     float operator[](int i) const { return *(&x + i); }
     float& operator[](int i) { return *(&x + i); }
 };



__device float3 make_float3(float x, float y, float z)
{
 #ifdef WITH_OPENCL
     float3 a = {x, y, z, 0.0f};
 #else
     float3 a = {x, y, z};
 #endif
     return a;
}

uint iDivUp(uint a, uint b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

class particle_tuple {
public:
  int i;
  int j;
  particle_tuple(int i, int j) {
    this->i = i;
    this->j = j;
  }
};

void save_file(BYTE *pixels, char *file_name, int height, int width) {

  FILE *imageFile;
  int x, y;
  BYTE pixel;
  imageFile = fopen("image.pgm", "wb");
  if (imageFile == NULL) {
    perror("ERROR: Cannot open output file");
    exit(EXIT_FAILURE);
  }

  fprintf(imageFile, "P5\n");                   // P5 filetype
  fprintf(imageFile, "%d %d\n", width, height); // dimensions
  fprintf(imageFile, "255\n");                  // Max pixel

  /* Now write a greyscale ramp */
  for (x = 0; x < height; x++) {
    for (y = 0; y < width; y++) {
      pixel = pixels[x * height + y];
      fputc(pixel, imageFile);
    }
  }

  fclose(imageFile);
}

// worry about destuctor later
std::vector<particle_tuple *> get_tuples(int n) {
  std::vector<particle_tuple *> tuples;
  if (n >= 2) {
    for (int i = 0; i < n; i++) {
      for (int j = i + 1; j < n; j++) {
        tuples.push_back(new particle_tuple(i, j));
      }
    }
  } else {
    printf("Error: only one particle\n");
  }
  printf("\n");
  return tuples;
}

float min(const float &a, const float &b) { return (a < b) ? a : b; }


float norm(const float2 &a) { return sqrtf(a.x * a.x + a.y * a.y); }

float distance(const float2 &a, const float2 &b) {
  return sqrtf((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

float distance_ground(const float2 &a, const float &rad,
                      const float &ground_y) {
  float res = ground_y - a.y - rad;
  return res;
}

float dot(const float2 &a, const float2 &b) { return a.x * b.x + a.y * b.y; }

void project_on_vector(const float2 &a, const float2 &b_normalized,
                       float2 &out) {
  float d = dot(a, b_normalized);
  out.x = b_normalized.x * d;
  out.y = b_normalized.y * d;
}

void clamp(float2 &v, float maxValue) {
  float lengthV = sqrtf(v.x * v.x + v.y * v.y);
  if (lengthV > maxValue) {
    float mult = (maxValue / lengthV);
    v.x *= mult;
    v.y *= mult;
  }
}

class Wall {
public:
  float2 x0;
  float2 x1;
  float2 n;
  float2 t;
  float2 t_norm;
  float a;
  float b;
  float c;
  float ab_sq;
  float ab_abs;
  float ac;
  float bc;
  float length;
  float width;
  Wall(float2 x0, float2 x1, float2 n) {
    this->x0 = x0;
    this->x1 = x1;
    this->n = n;
    this->t = make_float2(x1.x - x0.x, x1.y - x0.y);
    this->length = sqrtf(t.x * t.x + t.y * t.y);
    this->t_norm = make_float2(t.x / length, t.y / length);
    this->width = 0.05; // TODO fix later
    this->a = x1.y - x0.y;
    this->b = x0.x - x1.x;
    this->c = -(a * x0.x + b * x0.y);
    this->ab_sq = a * a + b * b;
    this->ab_abs = sqrtf(a * a + b * b);
    this->ac = a * c;
    this->bc = b * c;
  }
};

class Particle {
public:
  float2 X;
  float2 X_pred;
  float2 Delta_x;  // becomes Delta_buf
  int Delta_x_ctr; // becomes Delta_buf_ctr
  float2 V;
  float2 V_prev;
  float V_pref;
  float2 Accel;
  float mass;
  float inv_mass;
  int group;
  float2 goal;
  float r;
  // grid
  int cell_id;
  int cell_x;
  int cell_y;
  float3 color;

  Particle(float2 X, float2 V, float mass, float r, int group, float3 color,
           float2 goal) {
    this->X = X;
    this->X_pred = make_float2(X.x, X.y);
    this->Delta_x = make_float2(0., 0.);
    this->Delta_x_ctr = 0;
    this->V = V;
    this->Accel = make_float2(0., 0.);
    this->V_prev = make_float2(0., 0.);
    this->V_pref = V_PREF_ACCEL;
    this->mass = mass;
    this->inv_mass = 1.0 / mass;
    this->group = group;
    this->goal = goal;
    this->r = r;
    // TODO add cell_id, x, y for multiple grids
    this->cell_id = K_NOT_USED;
    this->cell_x = K_NOT_USED;
    this->cell_y = K_NOT_USED;
    this->color = color;
  }
};

class Simulation;
class Grid {
public:
  static const int max_per_cell = 10;
  int num_particles;
  int num_cells;
  int num_rows;
  int num_cols;
  float cell_size;
  float2 min_;
  float2 max_;
  int *grid_counters; // stores num of particles in each cell
  int **grid_cells;   // stores the particles indicies for each cell
                    // has a maximum number of particles per cell
  uint num_blocks;
  uint num_threads;

  Grid(int num_particles, float dummy_cell_size, float2 min_, float2 max_) {
    this->num_particles = num_particles;
    this->cell_size = dummy_cell_size;
    this->min_ = min_;
    this->max_ = max_;
    this->num_cells = (max_.x - min_.x) * (max_.y - min_.y);
    this->num_cols = (max_.x - min_.x) / cell_size;
    this->num_rows = (max_.y - min_.y) / cell_size;
    this->grid_counters = (int *)malloc(num_cells * (sizeof(int)));
    for (int i = 0; i < num_cells; i++) {
      this->grid_counters[i] = 0;
    }
    this->grid_cells = (int **)malloc(num_cells * (sizeof(int *)));
    for (int i = 0; i < num_cells; i++) {
      int *particle_indices = (int *)malloc(max_per_cell * (sizeof(int)));
      for (int j = 0; j < max_per_cell; j++) {
        particle_indices[j] = 0;
      }
      this->grid_cells[i] = particle_indices;
    }
    this->num_threads = min(BLOCK_SIZE, num_particles);
    this->num_blocks = iDivUp(num_particles, this->num_threads);
  }

  void update_stability(Particle **particles) // this is a kernel function
  {
    // reset\update grid counters
    for (int i = 0; i < num_cells; i++) {
      grid_counters[i] = 0;
      for (int j = 0; j < max_per_cell; j++) {
        grid_cells[i][j] = K_NOT_USED;
      }
    }

    // adding particles to grid
    for (int i = 0; i < num_particles; i++) {
      float2 X = particles[i]->X;
      int x = (X.x - min_.x) / cell_size;
      int y = (X.y - min_.y) / cell_size;
      int cell_id = y * num_rows + x;
      particles[i]->cell_id = cell_id;
      particles[i]->cell_x = x;
      particles[i]->cell_y = y;
      int tmp = grid_counters[cell_id];
      grid_cells[cell_id][tmp] = i;
      grid_counters[cell_id] += 1;
    }
  }

  bool is_colliding_stability(Particle **particles, int i, int j) const {

    float2 X = particles[i]->X;
    int xi = (X.x - min_.x) / cell_size;
    int yi = (X.y - min_.y) / cell_size;
    int cell_id_i = yi * num_rows + xi;
    X = particles[j]->X;
    int xj = (X.x - min_.x) / cell_size;
    int yj = (X.y - min_.y) / cell_size;
    int cell_id_j = yj * num_rows + xj;
    int is_x_neighbour = xi - xj;
    int is_y_neighbour = yi - yj;
    bool res = is_x_neighbour >= -3 && is_x_neighbour <= 3 &&
               is_y_neighbour >= -3 && is_y_neighbour <= 3;
    return res;
  }


  void update(Particle **particles) // this is a kernel function
  {
    // reset\update grid counters
    for (int i = 0; i < num_cells; i++) {
      grid_counters[i] = 0;
      for (int j = 0; j < max_per_cell; j++) {
        grid_cells[i][j] = K_NOT_USED;
      }
    }
    // adding particles to grid
    for (int i = 0; i < num_particles; i++) {
      float2 X = particles[i]->X_pred;
      int x = (X.x - min_.x) / cell_size;
      int y = (X.y - min_.y) / cell_size;
      int cell_id = y * num_rows + x;
      particles[i]->cell_id = cell_id;
      particles[i]->cell_x = x;
      particles[i]->cell_y = y;
      int tmp = grid_counters[cell_id];
      grid_cells[cell_id][tmp] = i;
      grid_counters[cell_id] += 1;
    }
  }

  /*two options:
    1) is colliding should be a 2d matrix that we preprocess
    in the update step
    then, is colliding just returns true\false based on that matrix
    2) have each particle loop around the surronding cells to see if they are
       colliding
   */
  bool is_colliding(Particle **particles, int i, int j) const {

    float2 X = particles[i]->X_pred;
    int xi = (X.x - min_.x) / cell_size;
    int yi = (X.y - min_.y) / cell_size;
    int cell_id_i = yi * num_rows + xi;
    X = particles[j]->X_pred;
    int xj = (X.x - min_.x) / cell_size;
    int yj = (X.y - min_.y) / cell_size;
    int cell_id_j = yj * num_rows + xj;
    int is_x_neighbour = xi - xj;
    int is_y_neighbour = yi - yj;
    bool res = is_x_neighbour >= -3 && is_x_neighbour <= 3 &&
               is_y_neighbour >= -3 && is_y_neighbour <= 3;
    return res;
  }

  ~Grid() {
    free(grid_counters);
    for (int i = 0; i < num_cells; i++) {
      free(grid_cells[i]);
    }
    free(grid_cells);
  }
};

class Constraint { // int i1,i2,... praticle indices
public:
  const Simulation *sim;
  int *indicies;
  int num_particles;
  float2 *delta_X;
  bool active;

  Constraint(Simulation *sim, int num_particles) {
    this->sim = sim;
    this->num_particles = num_particles;
    this->delta_X = (float2 *)malloc(num_particles * sizeof(float2));
    this->indicies = (int *)malloc(num_particles * (sizeof(int)));
    this->active = false;
    for (int i = 0; i < num_particles; i++) {
      delta_X[i] = make_float2(0., 0.);
    }
  }

  virtual void
  project(Particle **particles) = 0; // forcing implemntation in base class
  virtual ~Constraint() {
    free(indicies);
    free(delta_X);
  }
};

// should be constructed once for each scenerio
class PathPlanner {
public:
  int num_particles;
  Particle **particles;
  float2 *goals;
  float2 *velocity_buffer;
  PathPlanner(int num_particles, Particle **particles) {
    this->num_particles = num_particles;
    this->particles = particles;
    this->velocity_buffer = (float2 *)malloc(sizeof(float2) * num_particles);
    this->goals = (float2 *)malloc(sizeof(float2) * num_particles);
    for (int i = 0; i < num_particles; i++) {
      this->velocity_buffer[i] = make_float2(0., 0.);
      // this->goals[i]=make_float2(particles[i]->goal.x,particles[i]->goal.y);
    }
  }
  // TODO get current velocity, adjust predicted particle accordinfly for
  // smoothness

  void calc_pref_v_force(const int &particle_id) // returns velocity
  {
    const Particle *p = this->particles[particle_id];
    float2 goal = p->goal;
    this->velocity_buffer[particle_id].x = goal.x - p->X.x;
    this->velocity_buffer[particle_id].y = goal.y - p->X.y;
    const float length =
        sqrtf(velocity_buffer[particle_id].x * velocity_buffer[particle_id].x +
              velocity_buffer[particle_id].y * velocity_buffer[particle_id].y);
    if (length != 0) {
      this->velocity_buffer[particle_id].x /= length;
      this->velocity_buffer[particle_id].y /= length;
      this->velocity_buffer[particle_id].x *= p->V_pref;
      this->velocity_buffer[particle_id].y *= p->V_pref;
    }
  }

  void calc_velocity(const int &particle_id) // returns velocity
  {
    const Particle *p = this->particles[particle_id];
    // const float2 goal=p->goal;
    float2 goal = p->goal;
    // goal.x=p->X.x;
    // goal.y=GROUND_HEIGHT;
    this->velocity_buffer[particle_id].x = goal.x - p->X.x;
    this->velocity_buffer[particle_id].y = goal.y - p->X.y;
    const float length =
        sqrtf(velocity_buffer[particle_id].x * velocity_buffer[particle_id].x +
              velocity_buffer[particle_id].y * velocity_buffer[particle_id].y);
    if (length != 0) {
      this->velocity_buffer[particle_id].x /= length;
      this->velocity_buffer[particle_id].y /= length;
      this->velocity_buffer[particle_id].x *= p->V_pref;
      this->velocity_buffer[particle_id].y *= p->V_pref;
      // part below needs to be removed
      // add clamping here!
      /*
      this->velocity_buffer[particle_id].x=(1.0-KSI)*particles[particle_id]->V.x
          +KSI*velocity_buffer[particle_id].x;
      this->velocity_buffer[particle_id].y=(1.0-KSI)*particles[particle_id]->V.y
          +KSI*velocity_buffer[particle_id].y;
      */

      // clamping v between iterations
      /*
      float max_dv_mag = 0.08;
      float
      dv_x=this->velocity_buffer[particle_id].x-particles[particle_id]->V_prev.x;
      float
      dv_y=this->velocity_buffer[particle_id].y-particles[particle_id]->V_prev.y;
      float dv_mag=sqrt(dv_x*dv_x+dv_y*dv_y);
      if(dv_mag>max_dv_mag)
      {
          float mult = (max_dv_mag/dv_mag);
          this->velocity_buffer[particle_id].x*=mult;
          this->velocity_buffer[particle_id].y*=mult;
          //printf("%.3f %.3f\n",dv_mag,mult);
      }
      */
    }
  }
  ~PathPlanner() {
    free(velocity_buffer);
    free(goals);
  }
};

class Simulation {
public:
  int num_particles;
  int num_constraints;
  float time_step;
  Constraint **constraints;
  Particle **particles;
  PathPlanner *planner;
  Grid *grid;
  Grid *stability_grid;
  FILE *out;
  std::unordered_map<unsigned long long, Constraint *> collision_map;
  Constraint **collision_upper_trig_arr;
  Constraint **powerlaw_upper_trig_arr;
  Constraint **stability_upper_trig_arr;
  int step_no;
  float friction_constraint_stiffness;
  int num_walls;
  Wall **walls;

  Simulation(int num_particles, int num_constraints, float time_step,
             char *out_path) {
    this->num_particles = num_particles;
    this->time_step = time_step;
    this->particles = (Particle **)malloc(sizeof(void *) * num_particles);
    this->planner = new PathPlanner(num_particles, this->particles);
    this->out = fopen(out_path, "w");
    this->num_constraints = 0;
    this->constraints = NULL;
    this->collision_map =
        std::unordered_map<unsigned long long, Constraint *>();
    this->collision_upper_trig_arr = NULL;
    this->powerlaw_upper_trig_arr = NULL;
    this->stability_upper_trig_arr = NULL;
    this->grid = new Grid(num_particles, 2.66,
                          make_float2(LEFT_BOUND_X - 50, GRID_UP_BOUND - 10),
                          make_float2(RIGHT_BOUND_X + 50, GRID_LOW_BOND));
    /*
    this->stability_grid=new Grid(num_particles,2.16, //5.66, //7.66,
            make_float2(LEFT_BOUND_X-50,GRID_UP_BOUND-10),
            make_float2(RIGHT_BOUND_X+50,GRID_LOW_BOND));
    */
    this->stability_grid =
        new Grid(num_particles, 2.2, // 5.66, //7.66,
                 make_float2(LEFT_BOUND_X - 50, GRID_UP_BOUND - 10),
                 make_float2(RIGHT_BOUND_X + 50, GRID_LOW_BOND));
    this->step_no = 1;
    this->friction_constraint_stiffness = 0.22f;
    this->num_walls = 0;
    this->walls = NULL;
  }

  void calc_constraint_stiffness(int n) {
    // 1.-(1.-0.25)**(4./6)
    friction_constraint_stiffness =
        1.0f - powf(1.0f - friction_constraint_stiffness, (1.0f / n));
  }

  void stabilization() {
    stability_grid->update_stability(particles);

    for (int i = 0; i < 1; i++) {
      for (int i = 0; i < num_particles; i++) {
        particles[i]->Delta_x.x = 0.;
        particles[i]->Delta_x.y = 0.;
        particles[i]->Delta_x_ctr = 0;
      }

      // friction constraints
      for (int i = 0; i < num_particles; i++) {
        // iterate over adjacent cells
        for (int x = -1; x <= 1; x++) {
          int cur_x = particles[i]->cell_x + x;
          if (cur_x >= 0 && cur_x < stability_grid->num_cols) {
            for (int y = -1; y <= 1; y++) {
              int cur_y = particles[i]->cell_y + y;
              if (cur_y >= 0 && cur_y < stability_grid->num_rows) {
                int cell_id =
                    particles[i]->cell_id + x + (y * stability_grid->num_rows);
                if (stability_grid->grid_counters[cell_id] > 0) {
                  for (int idx = 0;
                       idx < stability_grid->grid_counters[cell_id]; idx++) {
                    int j = stability_grid->grid_cells[cell_id][idx];
                    if (i < j) // so only do collision once
                    {
                      int t_idx = (num_particles * i) + j - (i * (i + 1) * 0.5);
                      stability_upper_trig_arr[t_idx]->project(particles);
                    }
                  }
                }
              }
            }
          }
        }
      }

      // traverse friction constraints to accumalte deltas
      for (int i = 0; i < num_particles; i++) {
        // iterate over adjacent cells
        for (int x = -1; x <= 1; x++) {
          int cur_x = particles[i]->cell_x + x;
          if (cur_x >= 0 && cur_x < stability_grid->num_cols) {
            for (int y = -1; y <= 1; y++) {
              int cur_y = particles[i]->cell_y + y;
              if (cur_y >= 0 && cur_y < stability_grid->num_rows) {
                int cell_id =
                    particles[i]->cell_id + x + (y * stability_grid->num_rows);
                if (stability_grid->grid_counters[cell_id] > 0) {
                  for (int idx = 0;
                       idx < stability_grid->grid_counters[cell_id]; idx++) {
                    int j = stability_grid->grid_cells[cell_id][idx];
                    if (i < j) // so only do collision once
                    {
                      int t_idx = (num_particles * i) + j - (i * (i + 1) * 0.5);
                      if (stability_upper_trig_arr[t_idx]->active) {
                        for (int ctr = 0;
                             ctr <
                             stability_upper_trig_arr[t_idx]->num_particles;
                             ctr++) {
                          int p_idx =
                              stability_upper_trig_arr[t_idx]->indicies[ctr];
                          particles[p_idx]->Delta_x.x +=
                              stability_upper_trig_arr[t_idx]->delta_X[ctr].x;
                          particles[p_idx]->Delta_x.y +=
                              stability_upper_trig_arr[t_idx]->delta_X[ctr].y;
                          particles[p_idx]->Delta_x_ctr++;
                        }
                        stability_upper_trig_arr[t_idx]->active = false;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }

      for (int i = 0; i < num_particles; i++) {
        if (particles[i]->Delta_x_ctr > 0) {
          float dx =
              ALPHA * particles[i]->Delta_x.x / particles[i]->Delta_x_ctr;
          float dy =
              ALPHA * particles[i]->Delta_x.y / particles[i]->Delta_x_ctr;
          particles[i]->X_pred.x += dx;
          particles[i]->X_pred.y += dy;
          particles[i]->X.x += dx;
          particles[i]->X.y += dy;
        }
      }
    }
  }

  void project_velocity_constraints() {
    for (int i = 0; i < num_particles; i++) {
      particles[i]->Delta_x.x = 0.;
      particles[i]->Delta_x.y = 0.;
      particles[i]->Delta_x_ctr = 0;
    }

    for (int i = 0; i < num_particles; i++) {
      // iterate over adjacent cells
      for (int x = -2; x <= 2; x++) {
        int cur_x = particles[i]->cell_x + x;
        if (cur_x >= 0 && cur_x < grid->num_cols) {
          for (int y = -2; y <= 2; y++) {
            int cur_y = particles[i]->cell_y + y;
            if (cur_y >= 0 && cur_y < grid->num_rows) {
              int cell_id = particles[i]->cell_id + x + (y * grid->num_rows);
              if (grid->grid_counters[cell_id] > 0) {
                for (int idx = 0; idx < grid->grid_counters[cell_id]; idx++) {
                  int j = grid->grid_cells[cell_id][idx];
                  if (i < j) // so only do collision once
                  {
                    int t_idx = (num_particles * i) + j - (i * (i + 1) * 0.5);
                    powerlaw_upper_trig_arr[t_idx]->project(particles);
                  }
                }
              }
            }
          }
        }
      }
    }

    // traverse friction constraints to accumalte deltas
    for (int i = 0; i < num_particles; i++) {
      // iterate over adjacent cells
      for (int x = -2; x <= 2; x++) {
        int cur_x = particles[i]->cell_x + x;
        if (cur_x >= 0 && cur_x < grid->num_cols) {
          for (int y = -2; y <= 2; y++) {
            int cur_y = particles[i]->cell_y + y;
            if (cur_y >= 0 && cur_y < grid->num_rows) {
              int cell_id = particles[i]->cell_id + x + (y * grid->num_rows);
              if (grid->grid_counters[cell_id] > 0) {
                for (int idx = 0; idx < grid->grid_counters[cell_id]; idx++) {
                  int j = grid->grid_cells[cell_id][idx];
                  if (i < j) // so only do collision once
                  {
                    int t_idx = (num_particles * i) + j - (i * (i + 1) * 0.5);
                    if (powerlaw_upper_trig_arr[t_idx]->active) {
                      for (int ctr = 0;
                           ctr < powerlaw_upper_trig_arr[t_idx]->num_particles;
                           ctr++) {
                        int p_idx =
                            powerlaw_upper_trig_arr[t_idx]->indicies[ctr];
                        particles[p_idx]->Delta_x.x +=
                            powerlaw_upper_trig_arr[t_idx]->delta_X[ctr].x;
                        particles[p_idx]->Delta_x.y +=
                            powerlaw_upper_trig_arr[t_idx]->delta_X[ctr].y;
                        particles[p_idx]->Delta_x_ctr++;
                      }
                      powerlaw_upper_trig_arr[t_idx]->active = false;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    for (int i = 0; i < num_particles; i++) {
      particles[i]->V.x *= 0.99;
      particles[i]->V.y *= 0.99;

      float k = 1.; // 0.05; //stiffness changes with iteration;
      float2 dv_pref = particles[i]->V_prev;
      dv_pref.x = k * (planner->velocity_buffer[i].x - particles[i]->V.x);
      dv_pref.y = k * (planner->velocity_buffer[i].y - particles[i]->V.y);
      clamp(dv_pref, time_step * MAX_ACCEL);

      if (particles[i]->Delta_x_ctr > 0) {
        float dvx = (dv_pref.x + particles[i]->Delta_x.x) /
                    (1. + particles[i]->Delta_x_ctr);
        float dvy = (dv_pref.y + particles[i]->Delta_x.y) /
                    (1. + particles[i]->Delta_x_ctr);
        particles[i]->V.x += dvx;
        particles[i]->V.y += dvy;
      } else {
        particles[i]->V.x += dv_pref.x;
        particles[i]->V.y += dv_pref.y;
      }
      // clamp(particles[i]->V,MAX_SPEED);
      particles[i]->X_pred.x =
          particles[i]->X.x + particles[i]->V.x * time_step;
      particles[i]->X_pred.y =
          particles[i]->X.y + particles[i]->V.y * time_step;
      // perhaps clamp cannot be here, but rather in the constraints themselves
      // so to force that each constraint cannot become
      // TODO also need to clamp maximum speed change clamp
    }
  }

  void project_constraints() {

    for (int i = 0; i < num_particles; i++) {
      particles[i]->Delta_x.x = 0.;
      particles[i]->Delta_x.y = 0.;
      particles[i]->Delta_x_ctr = 0;
    }

    // friction constraints
    for (int i = 0; i < num_particles; i++) {
      // iterate over adjacent cells
      for (int x = -1; x <= 1; x++) {
        int cur_x = particles[i]->cell_x + x;
        if (cur_x >= 0 && cur_x < grid->num_cols) {
          for (int y = -1; y <= 1; y++) {
            int cur_y = particles[i]->cell_y + y;
            if (cur_y >= 0 && cur_y < grid->num_rows) {
              int cell_id = particles[i]->cell_id + x + (y * grid->num_rows);
              if (grid->grid_counters[cell_id] > 0) {
                for (int idx = 0; idx < grid->grid_counters[cell_id]; idx++) {
                  int j = grid->grid_cells[cell_id][idx];
                  if (i < j) // so only do collision once
                  {
                    // collision_map[i * num_particles + j]->project(particles);
                    int t_idx = (num_particles * i) + j - (i * (i + 1) * 0.5);
                    collision_upper_trig_arr[t_idx]->project(particles);
                    // powerlaw_upper_trig_arr[t_idx]->project(particles);
                    // stability_upper_trig_arr[t_idx]->project(particles);
                  }
                }
              }
            }
          }
        }
      }
    }
    // ground constraints
    for (int i = 0; i < num_constraints; i++) {
      constraints[i]->project(particles);
    }

    // traverse friction constraints to accumalte deltas
    for (int i = 0; i < num_particles; i++) {
      // iterate over adjacent cells
      for (int x = -1; x <= 1; x++) {
        int cur_x = particles[i]->cell_x + x;
        if (cur_x >= 0 && cur_x < grid->num_cols) {
          for (int y = -1; y <= 1; y++) {
            int cur_y = particles[i]->cell_y + y;
            if (cur_y >= 0 && cur_y < grid->num_rows) {
              int cell_id = particles[i]->cell_id + x + (y * grid->num_rows);
              if (grid->grid_counters[cell_id] > 0) {
                for (int idx = 0; idx < grid->grid_counters[cell_id]; idx++) {
                  int j = grid->grid_cells[cell_id][idx];
                  if (i < j) // so only do collision once
                  {
                    int t_idx = (num_particles * i) + j - (i * (i + 1) * 0.5);
                    if (collision_upper_trig_arr[t_idx]->active) {
                      for (int ctr = 0;
                           ctr < collision_upper_trig_arr[t_idx]->num_particles;
                           ctr++) {
                        int p_idx =
                            collision_upper_trig_arr[t_idx]->indicies[ctr];
                        particles[p_idx]->Delta_x.x +=
                            collision_upper_trig_arr[t_idx]->delta_X[ctr].x;
                        particles[p_idx]->Delta_x.y +=
                            collision_upper_trig_arr[t_idx]->delta_X[ctr].y;
                        particles[p_idx]->Delta_x_ctr++;
                      }
                      collision_upper_trig_arr[t_idx]->active = false;
                    }

                    /*
                    if(powerlaw_upper_trig_arr[t_idx]->active)
                    {
                        for(int ctr=0;
                            ctr<powerlaw_upper_trig_arr[t_idx]->num_particles
                            ;ctr++)
                        {
                            int
                    p_idx=powerlaw_upper_trig_arr[t_idx]->indicies[ctr];
                            particles[p_idx]->Delta_x.x +=
                    powerlaw_upper_trig_arr[t_idx]->delta_X[ctr].x;
                            particles[p_idx]->Delta_x.y +=
                    powerlaw_upper_trig_arr[t_idx]->delta_X[ctr].y;
                            particles[p_idx]->Delta_x_ctr++;
                        }
                        powerlaw_upper_trig_arr[t_idx]->active=false;
                    }
                    */
                  }
                }
              }
            }
          }
        }
      }
    }

    for (int i = 0; i < num_constraints; i++) {
      if (constraints[i]->active) {
        for (int j = 0; j < constraints[i]->num_particles; j++) {
          int idx = constraints[i]->indicies[j];
          particles[idx]->Delta_x.x += constraints[i]->delta_X[j].x;
          particles[idx]->Delta_x.y += constraints[i]->delta_X[j].y;
          particles[idx]->Delta_x_ctr++;
        }
        constraints[i]->active = false;
      }
    }

    for (int i = 0; i < num_particles; i++) {
      if (particles[i]->Delta_x_ctr > 0) {
        particles[i]->X_pred.x +=
            ALPHA * particles[i]->Delta_x.x / particles[i]->Delta_x_ctr;
        particles[i]->X_pred.y +=
            ALPHA * particles[i]->Delta_x.y / particles[i]->Delta_x_ctr;
        // clamp
        if (false) {

          float maxValue = 0.069;
          float length_d_i = distance(particles[i]->X_pred, particles[i]->X);
          if (length_d_i > maxValue) {
            float mult = (maxValue / length_d_i);
            particles[i]->X_pred.x =
                particles[i]->X.x +
                (particles[i]->X_pred.x - particles[i]->X.x) * mult;
            particles[i]->X_pred.y =
                particles[i]->X.y +
                (particles[i]->X_pred.y - particles[i]->X.y) * mult;
          }

        }
      }
    }
  }

  void do_time_step_force() {
    printf("Force Solve Frame %d\n", step_no);
    for (int i = 0; i < num_particles; i++) {
      planner->calc_pref_v_force(i);
      particles[i]->V_prev.x = planner->velocity_buffer[i].x;
      particles[i]->V_prev.y = planner->velocity_buffer[i].y;
    }

    // TODO change grid cell size
    stability_grid->update_stability(particles);
    // update grid by current positions

    // TODO calculate preffered speed
    //_vPref *= _prefSpeed/sqrtf(distSqToGoal);

    for (int i = 0; i < 1; i++) {
      for (int i = 0; i < num_particles; i++) {
        particles[i]->Delta_x.x = 0.;
        particles[i]->Delta_x.y = 0.;
        particles[i]->Delta_x_ctr = 0;
      }

      for (int i = 0; i < num_particles; i++) {
        // iterate over adjacent cells
        for (int x = -3; x <= 3; x++) {
          int cur_x = particles[i]->cell_x + x;
          if (cur_x >= 0 && cur_x < stability_grid->num_cols) {
            for (int y = -3; y <= 3; y++) {
              int cur_y = particles[i]->cell_y + y;
              if (cur_y >= 0 && cur_y < stability_grid->num_rows) {
                int cell_id =
                    particles[i]->cell_id + x + (y * stability_grid->num_rows);
                if (stability_grid->grid_counters[cell_id] > 0) {

                  for (int idx = 0;
                       idx < stability_grid->grid_counters[cell_id]; idx++) {
                    int j = stability_grid->grid_cells[cell_id][idx];
                    if (i < j) // so only do collision once
                    {

                      int t_idx = (num_particles * i) + j - (i * (i + 1) * 0.5);
                      // printf("Will project %d %d\n",i,j);
                      stability_upper_trig_arr[t_idx]->project(particles);
                    }
                  }
                }
              }
            }
          }
        }
      }

      for (int i = 0; i < num_particles; i++) {
        // iterate over adjacent cells
        for (int x = -3; x <= 3; x++) {
          int cur_x = particles[i]->cell_x + x;
          if (cur_x >= 0 && cur_x < stability_grid->num_cols) {
            for (int y = -3; y <= 3; y++) {
              int cur_y = particles[i]->cell_y + y;
              if (cur_y >= 0 && cur_y < stability_grid->num_rows) {
                int cell_id =
                    particles[i]->cell_id + x + (y * stability_grid->num_rows);
                if (stability_grid->grid_counters[cell_id] > 0) {
                  for (int idx = 0;
                       idx < stability_grid->grid_counters[cell_id]; idx++) {
                    int j = stability_grid->grid_cells[cell_id][idx];
                    if (i < j) // so only do collision once
                    {
                      int t_idx = (num_particles * i) + j - (i * (i + 1) * 0.5);
                      if (stability_upper_trig_arr[t_idx]->active) {
                        for (int ctr = 0;
                             ctr <
                             stability_upper_trig_arr[t_idx]->num_particles;
                             ctr++) {
                          int p_idx =
                              stability_upper_trig_arr[t_idx]->indicies[ctr];
                          particles[p_idx]->Delta_x.x +=
                              stability_upper_trig_arr[t_idx]->delta_X[ctr].x;
                          particles[p_idx]->Delta_x.y +=
                              stability_upper_trig_arr[t_idx]->delta_X[ctr].y;
                          particles[p_idx]->Delta_x_ctr++;
                        }
                        stability_upper_trig_arr[t_idx]->active = false;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }

      for (int i = 0; i < num_particles; i++) {
        particles[i]->Accel.x =
            (particles[i]->V_prev.x - particles[i]->V.x) / KSI_ACCEL;
        particles[i]->Accel.y =
            (particles[i]->V_prev.y - particles[i]->V.y) / KSI_ACCEL;

        if (particles[i]->Delta_x_ctr > 0) {
          /*
          define force constraint
            checks distance, tao.
            if distance < Nearest Neighbour
            calculate Delta_force for each particle
          */

          // accumalte and average delta forces for each particle
          // apply acceleration clamp
          // update velocity and positions
          // printf("In particles step %d\n",i);
          particles[i]->Accel.x +=
              particles[i]->Delta_x.x / particles[i]->Delta_x_ctr;
          particles[i]->Accel.y +=
              particles[i]->Delta_x.y / particles[i]->Delta_x_ctr;
          clamp(particles[i]->Accel, MAX_ACCEL);
        }
        particles[i]->V.x += particles[i]->Accel.x * time_step;
        particles[i]->V.y += particles[i]->Accel.y * time_step;
        printf("%d Speed %.4f\n", i,
               sqrtf(particles[i]->V.x * particles[i]->V.x +
                     particles[i]->V.y * particles[i]->V.y));
        particles[i]->X.x += particles[i]->V.x * time_step;
        particles[i]->X.y += particles[i]->V.y * time_step;
      }
    }

    step_no++;
  }

  void do_time_step() {
    printf("PBD Solve Frame %d\n", step_no);

    for (int i = 0; i < num_particles; i++) {
      planner->calc_velocity(i);

      particles[i]->V_prev.x = particles[i]->V.x;
      particles[i]->V_prev.y = particles[i]->V.y;

    }

    for (int i = 0; i < num_particles; i++) {
      particles[i]->X_pred.x += time_step * particles[i]->V.x;
      particles[i]->X_pred.y += time_step * particles[i]->V.y;
    }

    //----------------------stability grid stuff
    // stabilization();
    //-----------------------project constraints

    grid->update(particles);

    project_velocity_constraints();

    for (int i = 1; i < (ITER_COUNT + 1); i++) {
      calc_constraint_stiffness(i);
      project_constraints();
    }

    for (int i = 0; i < num_particles; i++) {
      float dx = particles[i]->X_pred.x - particles[i]->X.x;
      float dy = particles[i]->X_pred.y - particles[i]->X.y;
      particles[i]->V.x = dx / time_step;
      particles[i]->V.y = dy / time_step;
      particles[i]->X.x = particles[i]->X_pred.x;
      particles[i]->X.y = particles[i]->X_pred.y;
    }
    step_no++;
  }

  ~Simulation() {
    fclose(this->out);
    for (int i = 0; i < num_particles; i++) {
      delete particles[i];
    }
    for (int i = 0; i < num_constraints; i++) {
      delete constraints[i];
    }

    /*
    for (std::unordered_map<unsigned long long,
            Constraint*>::const_iterator it = collision_map.begin();
            it != collision_map.end(); ++it)
    {
       delete it->second;
    }
    */
    if (walls != NULL) {
      for (int i = 0; i < num_walls; i++) {
        delete walls[i];
      }
    }

    int trig_len = 1 + (num_particles * (num_particles + 1) / 2);
    for (int i = 0; i < num_particles; i++) {
      for (int j = 0; j < num_particles; j++) {
        if (i < j) {
          int r = i;
          int c = j;
          int t_idx = (num_particles * r) + c - (r * (r + 1) * 0.5);
          if (collision_upper_trig_arr != NULL) {
            delete collision_upper_trig_arr[t_idx];
          }
          if (powerlaw_upper_trig_arr != NULL) {
            delete powerlaw_upper_trig_arr[t_idx];
          }
          if (stability_upper_trig_arr != NULL) {
            delete stability_upper_trig_arr[t_idx];
          }
        }
      }
    }

    free(constraints);
    free(particles);
    free(collision_upper_trig_arr);
    free(powerlaw_upper_trig_arr);
    delete planner;
    delete grid;
  }
};

class Stability_Constraint : public Constraint {
public:
  int i;
  int j;
  float w_i_coef;
  float w_j_coef;
  float2 contact_normal;
  float2 tangential_displacement;
  float2 x_pred_w_delta1;
  float2 x_pred_w_delta2;
  float2 out;
  float collision_margin;

  Stability_Constraint(Simulation *sim, int i, int j) : Constraint(sim, 2) {
    this->i = i;
    this->j = j;
    this->indicies[0] = i;
    this->indicies[1] = j;
    // TODO if seg fault happens, it is because the particles are set up after
    // the constraints
    this->w_i_coef =
        sim->particles[i]->inv_mass /
        (sim->particles[i]->inv_mass + sim->particles[j]->inv_mass);
    this->w_j_coef =
        -sim->particles[j]->inv_mass /
        (sim->particles[i]->inv_mass + sim->particles[j]->inv_mass);
    this->collision_margin =
        (sim->particles[i]->r + sim->particles[j]->r) * 1.05f;
    this->contact_normal = make_float2(0.0, 0.0);
    this->tangential_displacement = make_float2(0.0, 0.0);
    this->x_pred_w_delta1 = make_float2(0., 0.);
    this->x_pred_w_delta2 = make_float2(0., 0.);
    this->out = make_float2(0., 0.);
  }

  virtual void project(Particle **particles) {
    // we don't want to use the bad old values
    delta_X[0].x = 0.0;
    delta_X[0].y = 0.0;
    delta_X[1].x = 0.0;
    delta_X[1].y = 0.0;
    if (true) {
      float d = distance(particles[i]->X, particles[j]->X);
      float f = d - collision_margin;
      if (f < 0) {
        contact_normal.x = 0.;
        contact_normal.y = 0.;
        tangential_displacement.x = 0.;
        tangential_displacement.y = 0.;
        x_pred_w_delta1.x = 0.;
        x_pred_w_delta1.y = 0.;
        x_pred_w_delta2.x = 0.;
        x_pred_w_delta2.y = 0.;
        out.x = 0.;
        out.y = 0.;
        contact_normal.x = (particles[i]->X.x - particles[j]->X.x) / d;
        contact_normal.y = (particles[i]->X.y - particles[j]->X.y) / d;
        delta_X[0].x = -w_i_coef * contact_normal.x * f;
        delta_X[0].y = -w_i_coef * contact_normal.y * f;
        delta_X[1].x = -w_j_coef * contact_normal.x * f;
        delta_X[1].y = -w_j_coef * contact_normal.y * f;
        active = true;
      }
    }
  }
};

class Powerlaw_Force_Constraint : public Constraint {
public:
  static const float k;// = 1.5f; // stiffness
  static const float m;// = 2.0f;
  static const float tao0;// = 3.f;
  int i;
  int j;
  float w_i_coef;
  float w_j_coef;
  float2 out;
  float collision_margin;
  float radius_init;
  float radius_sq_init;
  float delta_t;
  float dv_i;
  float dv_j;

  Powerlaw_Force_Constraint(Simulation *sim, int i, int j)
      : Constraint(sim, 2) {
    this->i = i;
    this->j = j;
    this->indicies[0] = i;
    this->indicies[1] = j;
    this->w_i_coef = sim->particles[i]->inv_mass;
    this->w_j_coef = sim->particles[j]->inv_mass;
    this->out = make_float2(0., 0.);
    this->collision_margin =
        (sim->particles[i]->r + sim->particles[j]->r) * 1.05f;
    this->radius_init = (sim->particles[i]->r + sim->particles[j]->r);
    this->radius_sq_init = radius_init * radius_init;
    this->delta_t = sim->time_step;
    this->dv_i = 1. / delta_t;
    this->dv_j = -1. / delta_t;
  }

  virtual void project(Particle **particles) {
    // we don't want to use the bad old values
    delta_X[0].x = 0.0;
    delta_X[0].y = 0.0;
    delta_X[1].x = 0.0;
    delta_X[1].y = 0.0;
    float2 x_i = particles[i]->X;
    float2 x_j = particles[j]->X;
    const float dist = distance(particles[i]->X, particles[j]->X);
    float radius_sq = radius_sq_init;
    if (dist < radius_init) {
      radius_sq = (radius_init - dist) * (radius_init - dist);
    }
    /*
    float v_x=(particles[i]->X_pred.x-x_i.x)/delta_t
                -(particles[j]->X_pred.x-x_j.x)/delta_t;
    float v_y=(particles[i]->X_pred.y-x_i.y)/delta_t
                -(particles[j]->X_pred.y-x_j.y)/delta_t;
        */
    float v_x = particles[i]->V.x - particles[j]->V.x;
    float v_y = particles[i]->V.y - particles[j]->V.y;
    float x0 = x_i.x - x_j.x;
    float y0 = x_i.y - x_j.y;
    float v_sq = v_x * v_x + v_y * v_y;
    float x0_sq = x0 * x0;
    float y0_sq = y0 * y0;
    float x_sq = x0_sq + y0_sq;
    float a = v_sq;
    float b = -v_x * x0 - v_y * y0;
    float b_sq = b * b;
    float c = x_sq - radius_sq;
    float d_sq = b_sq - a * c;
    if (d_sq > 0 && (a < -_EPSILON || a > _EPSILON)) {
      float d = sqrtf(d_sq);
      float tao = (b - d) / a;
      if (dist < NN_ACCEL && tao > 0) {
        float c_x_nom = (v_sq * x0 + b * v_x) / d;
        float c_x = v_x - c_x_nom;
        float c_y_nom = (v_sq * y0 + b * v_y) / d;
        float c_y = v_y - c_y_nom;
        float F_s =
            -k * exp(-tao / tao0) / (a * powf(tao, m)) * (m / tao + 1. / tao0);
        float F_x = c_x * F_s;
        float F_y = c_y * F_s;
        delta_X[0].x = F_x;
        delta_X[0].y = F_y;
        delta_X[1].x = -F_x;
        delta_X[1].y = -F_y;
        active = true;
      }
    }
  }
};
const float Powerlaw_Force_Constraint::k = 1.5f; // stiffness
const float Powerlaw_Force_Constraint::m = 2.0f;
const float Powerlaw_Force_Constraint::tao0 = 3.f;


class Powerlaw_Constraint : public Constraint {
public:
  static const float k;// = 1.5; // stiffness
  static const float tao0;// = 4.;
  static const float maxValue;// = 0.2; // delta_t * pref_speed
  int i;
  int j;
  float w_i_coef;
  float w_j_coef;
  float2 out;
  float collision_margin;
  float radius_init;
  float radius_sq_init;
  float delta_t;
  float dv_i;
  float dv_j;
  float max_acceleration;

  Powerlaw_Constraint(Simulation *sim, int i, int j) : Constraint(sim, 2) {
    this->i = i;
    this->j = j;
    this->indicies[0] = i;
    this->indicies[1] = j;
    // TODO if seg fault happens, it is because the particles are set up after
    // the constraints
    this->w_i_coef = sim->particles[i]->inv_mass;
    this->w_j_coef = sim->particles[j]->inv_mass;
    this->out = make_float2(0., 0.);
    this->collision_margin =
        (sim->particles[i]->r + sim->particles[j]->r) * 1.05f;
    this->radius_init = (sim->particles[i]->r + sim->particles[j]->r);
    this->radius_sq_init = radius_init * radius_init;
    this->delta_t = sim->time_step;
    this->dv_i = 1.;  // 1./delta_t;
    this->dv_j = -1.; //-1./delta_t;
    this->max_acceleration = sim->time_step * MAX_ACCEL;
  }

  virtual void project(Particle **particles) {
    // we don't want to use the bad old values
    delta_X[0].x = 0.0;
    delta_X[0].y = 0.0;
    delta_X[1].x = 0.0;
    delta_X[1].y = 0.0;

    float2 x_i = particles[i]->X;
    float2 x_j = particles[j]->X;
    const float dist = distance(particles[i]->X, particles[j]->X);
    float radius_sq = radius_sq_init;
    if (dist < radius_init) {
      radius_sq = (radius_init - dist) * (radius_init - dist);
    }
    float v_x = (particles[i]->X_pred.x - x_i.x) / delta_t -
                (particles[j]->X_pred.x - x_j.x) / delta_t;
    float v_y = (particles[i]->X_pred.y - x_i.y) / delta_t -
                (particles[j]->X_pred.y - x_j.y) / delta_t;
    float x0 = x_i.x - x_j.x;
    float y0 = x_i.y - x_j.y;
    float v_sq = v_x * v_x + v_y * v_y;
    float x0_sq = x0 * x0;
    float y0_sq = y0 * y0;
    float x_sq = x0_sq + y0_sq;
    float a = v_sq;
    float b = -v_x * x0 - v_y * y0;
    float b_sq = b * b;
    float c = x_sq - radius_sq;
    float d_sq = b_sq - a * c;
    if (false && d_sq > 0 && (a < -_EPSILON || a > _EPSILON)) {
      float d = sqrtf(d_sq);
      float tao = (b - d) / a;
      float tao_alt = (b + d) / a;
      // pick the min solution that is > 0
      tao = tao_alt < tao && tao_alt > 0 ? tao_alt : tao;
      // need to consider +- sign perhaps?
      if (tao > 0 /* && tao<tao0 */) {
        float clamp_tao = exp(-tao * tao / tao0);
        float c_tao = abs(tao - tao0);
        float tao_sq = c_tao * c_tao;
        float grad_x_i =
            2 * c_tao *
            ((dv_i / a) *
             ((-2. * v_x * tao) -
              (x0 + (v_y * x0 * y0 + v_x * (radius_sq - y0_sq)) / d)));
        float grad_y_i =
            2 * c_tao *
            ((dv_i / a) *
             ((-2. * v_y * tao) -
              (y0 + (v_x * x0 * y0 + v_y * (radius_sq - x0_sq)) / d)));
        float grad_x_j = -grad_x_i;
        float grad_y_j = -grad_y_i;
        float stiff = exp(-tao * tao / tao0);
        float s = 0.5 * tao_sq /
                  (particles[i]->inv_mass *
                       (grad_y_i * grad_y_i + grad_x_i * grad_x_i) +
                   particles[j]->inv_mass *
                       (grad_y_j * grad_y_j + grad_x_j * grad_x_j));
        active = true;
        delta_X[0].x = s * w_i_coef * grad_x_i;
        delta_X[0].y = s * w_i_coef * grad_y_i;
        clamp(delta_X[0], max_acceleration);
        delta_X[1].x = s * w_j_coef * grad_x_j;
        delta_X[1].y = s * w_j_coef * grad_y_j;
        clamp(delta_X[1], max_acceleration);
      }

      if (false && tao > 0) {
        float clamp_tao = exp(-tao * tao / tao0);
        float c_x_nom = (v_sq * x0 + b * v_x) / d;
        float c_x = v_x - c_x_nom;
        float c_y_nom = (v_sq * y0 + b * v_y) / d;
        float c_y = v_y - c_y_nom;
        float grad_x_i = dv_i + (v_y * y0 * dv_i / d) +
                         ((v_y * x0 * y0 + (radius_sq - y0_sq) * v_x) * dv_i *
                          c_x_nom / d_sq);
        float grad_y_i = dv_i + (v_x * x0 * dv_i / d) +
                         ((v_x * x0 * y0 + (radius_sq - x0_sq) * v_y) * dv_i *
                          c_y_nom / d_sq);
        float grad_x_j = dv_j + (v_y * y0 * dv_j / d) +
                         ((v_y * x0 * y0 + (radius_sq - y0_sq) * v_x) * dv_j *
                          c_x_nom / d_sq);
        float grad_y_j = dv_j + (v_x * x0 * dv_j / d) +
                         ((v_x * x0 * y0 + (radius_sq - x0_sq) * v_y) * dv_j *
                          c_y_nom / d_sq);
        grad_x_i *= 2 * abs(c_x);
        grad_y_i *= 2 * abs(c_y);
        grad_x_j *= 2 * abs(c_x);
        grad_y_j *= 2 * abs(c_y);
        c_x *= c_x;
        c_y *= c_y;
        float s_fin = c_x + c_y;
        float stiff = ((2. / tao) + (1. / tao0)) * k * exp(-tao / tao0) /
                      (v_sq * tao * tao);
        // float s_x=stiff*c_x/(particles[i]->inv_mass*grad_x_i*grad_x_i
        //              +particles[j]->inv_mass*grad_x_j*grad_x_j);
        // float s_y=stiff*c_y/(particles[i]->inv_mass*grad_y_i*grad_y_i
        //              +particles[j]->inv_mass*grad_y_j*grad_y_j);
        // s_x*=s_x;
        // s_y*=s_y;
        stiff = exp(-tao * tao / tao0);
        float s = 0.39 * s_fin /
                  (particles[i]->inv_mass *
                       (grad_y_i * grad_y_i + grad_x_i * grad_x_i) +
                   particles[j]->inv_mass *
                       (grad_y_j * grad_y_j + grad_x_j * grad_x_j));
        // grad_y_i=%f\n",stiff,s,grad_x_i,grad_y_i);
        active = true;
        delta_X[0].x = s * w_i_coef * grad_x_i;
        delta_X[0].y = s * w_i_coef * grad_y_i;
        clamp(delta_X[0], max_acceleration);
        delta_X[1].x = s * w_j_coef * grad_x_j;
        delta_X[1].y = s * w_j_coef * grad_y_j;
        clamp(delta_X[1], max_acceleration);
      }
    }
  }
};
const float Powerlaw_Constraint::k = 1.5; // stiffness
const float Powerlaw_Constraint::tao0 = 4.;
const float Powerlaw_Constraint::maxValue = 0.2; // delta_t * pref_speed


class Friction_Constraint : public Constraint {
public:
  // usually easier to keep object moving than start movement, so mu_s>mu_k
  // some friction results
  // http://hypertextbook.com/facts/2007/TabraizRasul.shtml
  static const float mui_static;// = 0.00026;    // 0.021;
  static const float mui_kinematic;// = 0.00023; // 0.02;
  /*typical values:
  http://spiff.rit.edu/classes/phys211/lectures/fric/fric_all.html smooth 0.05
  medium 0.3
  rough 1.0
  */
  int i;
  int j;
  float w_i_coef;
  float w_j_coef;
  float2 contact_normal;
  float2 tangential_displacement;
  float2 x_pred_w_delta1;
  float2 x_pred_w_delta2;
  float2 out;
  float collision_margin;
  float radius_sq_init;
  float radius_init;
  float delta_t;

  Friction_Constraint(Simulation *sim, int i, int j) : Constraint(sim, 2) {
    this->i = i;
    this->j = j;
    this->indicies[0] = i;
    this->indicies[1] = j;
    // TODO if seg fault happens, it is because the particles are set up after
    // the constraints
    this->w_i_coef =
        sim->particles[i]->inv_mass /
        (sim->particles[i]->inv_mass + sim->particles[j]->inv_mass);
    this->w_j_coef =
        -sim->particles[j]->inv_mass /
        (sim->particles[i]->inv_mass + sim->particles[j]->inv_mass);
    this->collision_margin =
        (sim->particles[i]->r + sim->particles[j]->r) * 1.05f;
    this->contact_normal = make_float2(0.0, 0.0);
    this->tangential_displacement = make_float2(0.0, 0.0);
    this->x_pred_w_delta1 = make_float2(0., 0.);
    this->x_pred_w_delta2 = make_float2(0., 0.);
    this->out = make_float2(0., 0.);
    this->radius_init = (sim->particles[i]->r + sim->particles[j]->r);
    this->radius_sq_init = radius_init * radius_init;
    this->delta_t = sim->time_step;
  }

  virtual void project(Particle **particles) {
    // we don't want to use the bad old values
    delta_X[0].x = 0.0;
    delta_X[0].y = 0.0;
    delta_X[1].x = 0.0;
    delta_X[1].y = 0.0;
    float d = distance(particles[i]->X_pred, particles[j]->X_pred);
    float f = d - collision_margin;
    if (f < 0) {
      contact_normal.x = 0.;
      contact_normal.y = 0.;
      tangential_displacement.x = 0.;
      tangential_displacement.y = 0.;
      x_pred_w_delta1.x = 0.;
      x_pred_w_delta1.y = 0.;
      x_pred_w_delta2.x = 0.;
      x_pred_w_delta2.y = 0.;
      out.x = 0.;
      out.y = 0.;
      contact_normal.x = (particles[i]->X_pred.x - particles[j]->X_pred.x) / d;
      contact_normal.y = (particles[i]->X_pred.y - particles[j]->X_pred.y) / d;
      delta_X[0].x = -w_i_coef * contact_normal.x * f;
      delta_X[0].y = -w_i_coef * contact_normal.y * f;
      delta_X[1].x = -w_j_coef * contact_normal.x * f;
      delta_X[1].y = -w_j_coef * contact_normal.y * f;
      x_pred_w_delta1.x = delta_X[0].x + particles[i]->X_pred.x;
      x_pred_w_delta1.y = delta_X[0].y + particles[i]->X_pred.y;
      x_pred_w_delta2.x = delta_X[1].x + particles[j]->X_pred.x;
      x_pred_w_delta2.y = delta_X[1].y + particles[j]->X_pred.y;
      float n_norm = distance(x_pred_w_delta1, x_pred_w_delta2);
      contact_normal.y = (x_pred_w_delta1.x - x_pred_w_delta2.x) / n_norm;
      contact_normal.x = -(x_pred_w_delta1.y - x_pred_w_delta2.y) / n_norm;
      // tangential_displacement.x = x_pred_w_delta1.x-x_pred_w_delta2.x;
      // tangential_displacement.y = x_pred_w_delta1.y-x_pred_w_delta2.y;
      // Above might be wrong
      // should be
      tangential_displacement.x = x_pred_w_delta1.x - particles[i]->X.x -
                                  (x_pred_w_delta2.x - particles[j]->X.x);
      tangential_displacement.y = x_pred_w_delta1.y - particles[i]->X.y -
                                  (x_pred_w_delta2.y - particles[j]->X.y);
      project_on_vector(tangential_displacement, contact_normal, out);
      float out_norm = norm(out);
      if (out_norm >= mui_static * d) {
        float coef = min(1., mui_kinematic * d / out_norm);
        out.x *= coef;
        out.y *= coef;
      }
      delta_X[0].x += -out.x * w_i_coef;
      delta_X[0].y += -out.y * w_i_coef;
      delta_X[1].x += -out.x * w_j_coef;
      delta_X[1].y += -out.y * w_j_coef;
      active = true;
    } else {
      float2 x_i = particles[i]->X;
      float2 x_j = particles[j]->X;
      const float dist = distance(particles[i]->X, particles[j]->X);
      float radius_sq = radius_sq_init;
      if (dist < radius_init) {
        radius_sq = (radius_init - dist) * (radius_init - dist);
      }
      const float v_ix = (particles[i]->X_pred.x - x_i.x) / delta_t;
      const float v_jx = (particles[j]->X_pred.x - x_j.x) / delta_t;
      const float v_x = v_ix - v_jx;
      const float v_iy = (particles[i]->X_pred.y - x_i.y) / delta_t;
      const float v_jy = (particles[j]->X_pred.y - x_j.y) / delta_t;
      const float v_y = v_iy - v_jy;
      float x0 = x_i.x - x_j.x;
      float y0 = x_i.y - x_j.y;
      float v_sq = v_x * v_x + v_y * v_y;
      float x0_sq = x0 * x0;
      float y0_sq = y0 * y0;
      float x_sq = x0_sq + y0_sq;
      float a = v_sq;
      float b = -v_x * x0 - v_y * y0;
      float b_sq = b * b;
      float c = x_sq - radius_sq;
      float d_sq = b_sq - a * c;
      if (d_sq > 0 && (a < -_EPSILON || a > _EPSILON)) {
        float d = sqrtf(d_sq);
        float tao = (b - d) / a;
        float tao_alt = (b + d) / a;
        // pick the min solution that is > 0
        tao = tao_alt < tao && tao_alt > 0 ? tao_alt : tao;
        // need to consider +- sign perhaps?
        if (tao > 0) {
          // const float min_tao_init=b/v_sq;
          const float min_tao =
              tao + delta_t; // min_tao_init;//(min_tao_init+tao)/2;
          const float x_i_min = x_i.x + min_tao * v_ix;
          const float y_i_min = x_i.y + min_tao * v_iy;
          const float x_j_min = x_j.x + min_tao * v_jx;
          const float y_j_min = x_j.y + min_tao * v_jy;
          float min_tao_dist = sqrtf((x_i_min - x_j_min) * (x_i_min - x_j_min) +
                                     (y_i_min - y_j_min) * (y_i_min - y_j_min));
          float d = min_tao_dist;
          float f = d - collision_margin;
          if (f < 0 && d > _EPSILON) {
            const float clamp_tao = exp(-min_tao * min_tao / 5.);
            const float k = sim->friction_constraint_stiffness; // 0.25;
            contact_normal.x = 0.;
            contact_normal.y = 0.;
            tangential_displacement.x = 0.;
            tangential_displacement.y = 0.;
            x_pred_w_delta1.x = 0.;
            x_pred_w_delta1.y = 0.;
            x_pred_w_delta2.x = 0.;
            x_pred_w_delta2.y = 0.;
            out.x = 0.;
            out.y = 0.;
            contact_normal.x = (x_i_min - x_j_min) / d;
            contact_normal.y = (y_i_min - y_j_min) / d;
            delta_X[0].x = -k * clamp_tao * w_i_coef * contact_normal.x * f;
            delta_X[0].y = -k * clamp_tao * w_i_coef * contact_normal.y * f;
            delta_X[1].x = -k * clamp_tao * w_j_coef * contact_normal.x * f;
            delta_X[1].y = -k * clamp_tao * w_j_coef * contact_normal.y * f;
            active = true;

            const float x_i_tao = x_i.x + tao * v_ix;
            const float y_i_tao = x_i.y + tao * v_iy;
            const float x_j_tao = x_j.x + tao * v_jx;
            const float y_j_tao = x_j.y + tao * v_jy;
            x_pred_w_delta1.x = delta_X[0].x + x_i_min;
            x_pred_w_delta1.y = delta_X[0].y + y_i_min;
            x_pred_w_delta2.x = delta_X[1].x + x_j_min;
            x_pred_w_delta2.y = delta_X[1].y + y_j_min;
            float n_norm = distance(x_pred_w_delta1, x_pred_w_delta2);
            contact_normal.y = (x_pred_w_delta1.x - x_pred_w_delta2.x) / n_norm;
            contact_normal.x =
                -(x_pred_w_delta1.y - x_pred_w_delta2.y) / n_norm;
            tangential_displacement.x =
                x_pred_w_delta1.x - x_i_tao - (x_pred_w_delta2.x - x_j_tao);
            tangential_displacement.y =
                x_pred_w_delta1.y - y_i_tao - (x_pred_w_delta2.y - y_j_tao);
            project_on_vector(tangential_displacement, contact_normal, out);
            float out_norm = norm(out);
            if (out_norm >= mui_static * d) {
              float coef = min(1., mui_kinematic * d / out_norm);
              out.x *= coef;
              out.y *= coef;
            }
            delta_X[0].x += -out.x * w_i_coef;
            delta_X[0].y += -out.y * w_i_coef;
            delta_X[1].x += -out.x * w_j_coef;
            delta_X[1].y += -out.y * w_j_coef;
            active = true;
          }
        }
      }
    }
  }
};
const float Friction_Constraint::mui_static = 0.00026;    // 0.021;
const float Friction_Constraint::mui_kinematic = 0.00023; // 0.02;


class Wall_Constraint : public Constraint {
public:
  // static const float ground_height=GROUND_HEIGHT;
  int i;
  // F=mui Fn = mui mg .
  // static const float
  // kinematic_friction=30.7*9.81*MS_PER_UPDATE*MS_PER_UPDATE;
  static const float kinematic_friction;// =  30000.7 * 9.81 * MS_PER_UPDATE * MS_PER_UPDATE;
  int wall_idx;
  float collision_margin;
  float2 contact_normal;
  Wall_Constraint(Simulation *sim, int i, int wall_idx) : Constraint(sim, 1) {
    this->i = i;
    this->indicies[0] = i;
    this->wall_idx = wall_idx;
    this->collision_margin =
        (sim->particles[i]->r + sim->walls[wall_idx]->width) * 1.05;
    this->contact_normal = make_float2(0., 0.);
  }

  virtual void project(Particle **particles) {
    delta_X[0].x = 0.0;
    delta_X[0].y = 0.0;
    Wall *wall = sim->walls[wall_idx];
    float2 p = particles[i]->X_pred;
    float2 p_prev = particles[i]->X;
    const float x_hit =
        (wall->b * (wall->b * p.x - wall->a * p.y) - wall->ac) / wall->ab_sq;
    // check if between x0,x1
    const float chk = (x_hit - wall->x0.x) / wall->t.x;
    if (chk <= 1 && chk >= 0) {
      // float d = distance(particles[i]->X, particles[j]->X);
      // float f = d - collision_margin;
      const float y_hit =
          (wall->a * (-wall->b * p.x + wall->a * p.y) - wall->bc) / wall->ab_sq;
      // const float s=a*p.x+b*p.y+c;
      const float d =
          abs(wall->a * p.x + wall->b * p.y + wall->c) / wall->ab_abs;
      float f = d - collision_margin;
      if (f < 0) {
        contact_normal.x = (p.x - x_hit) / d;
        contact_normal.y = (p.y - y_hit) / d;
        delta_X[0].x = -contact_normal.x * f;
        delta_X[0].y = -contact_normal.y * f;
        active = true;
      }
      /*
      vec2 dir = start - end;
      float lngth = length(dir);
      dir /= lngth;
      vec2 proj = max(0.0, min(lngth, dot((start - p), dir))) * dir;
      return length( (start - p) - proj ) - (width / 2.0);
      */
      /*
      const float
      y_hit=(wall->a*(-wall->b*p.x+wall->a*p.y)-wall->bc)/wall->ab_sq;
      //const float s=a*p.x+b*p.y+c;
      //const float dist=abs(s)/ab_abs;
      //TODO check if ray p_prev-p lies inside object
      const float c_p=(p_prev.x-x_hit)*wall->n.x+(p_prev.y-y_hit)*wall->n.y;
      if(c_p<0)
      {
          float s=c_p/(wall->n.x*wall->n.x+wall->n.y*wall->n.y);
          //project (figure out how to add distance later
          delta_X[0].x=-wall->n.x*s;
          delta_X[0].y=-wall->n.y*s;
          active=false;
      }
      */
    }
  }
};
const float Wall_Constraint::kinematic_friction =  30000.7 * 9.81 * MS_PER_UPDATE * MS_PER_UPDATE;



// TODO
class Ground_Constraint : public Constraint {
public:
  // static const float ground_height=GROUND_HEIGHT;
  int i;
  // F=mui Fn = mui mg .
  // static const float
  // kinematic_friction=30.7*9.81*MS_PER_UPDATE*MS_PER_UPDATE;
  static const float kinematic_friction; //=30000.7 * 9.81 * MS_PER_UPDATE * MS_PER_UPDATE;

  Ground_Constraint(Simulation *sim, int i) : Constraint(sim, 1) {
    this->i = i;
    this->indicies[0] = i;
  }

  virtual void project(Particle **particles) {
    // we don't want to use the bad old values
    delta_X[0].x = 0.0;
    delta_X[0].y = 0.0;
    float f =
        distance_ground(particles[i]->X_pred, particles[i]->r, GROUND_HEIGHT);
    if (f < 0) {
      // particles[i]->Delta_x_ctr+=1;            //TODO remove
      float x_d = particles[i]->X_pred.x - particles[i]->X.x;
      if (x_d > kinematic_friction)
      // if(particles[i]->V.x>kinematic_friction)
      {
        delta_X[0].x = -kinematic_friction;
      } else {
        delta_X[0].x = -x_d;
      }

      delta_X[0].y = f;
      active = true;
    }
  }
};
const float Ground_Constraint::kinematic_friction =
     30000.7 * 9.81 * MS_PER_UPDATE * MS_PER_UPDATE;


//----------------------------------------

double GetSeconds() {
  // Figure out time elapsed since last call to idle function
  static struct timeval last_idle_time;
  static double time = 0.0;
  struct timeval time_now;
  gettimeofday(&time_now, NULL);
  if (last_idle_time.tv_usec == 0)
    last_idle_time = time_now;
  float dt = (float)(time_now.tv_sec - last_idle_time.tv_sec) +
             1.0e-6 * (time_now.tv_usec - last_idle_time.tv_usec);
  time += dt;
  last_idle_time = time_now;
  return time;
}

static void error_callback(int error, const char *description) {
  fprintf(stderr, "Error %d: %s\n", error, description);
}

void set_camera(int display_w, int display_h, double rotate_camera) {
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(36, (double)display_w / (double)display_h, 0.1, 3000);
  gluLookAt(0, 180, 50, 0, 0, 0, 0, 1, 0);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
}

void drawCircle(float cx, float cy, float r, int num_segments, float3 color) {
  glColor3f((GLfloat)color.x, (GLfloat)color.y, (GLfloat)color.z);
  glBegin(GL_POLYGON);
  for (int ii = 0; ii < num_segments; ii++) {
    float theta = 2.0f * 3.1635926f * float(ii) /
                  float(num_segments); // get the current angle
    float x = r * cosf(theta);         // calculate the x component
    float y = r * sinf(theta);         // calculate the y component
    glVertex3f(x + cx, 2., y + cy);    // output vertex
  }
  glEnd();
}

void drawDirection(float cx, float cy, float dirx, float diry) {
  glColor3f((GLfloat)0.0, (GLfloat)0.0, (GLfloat)0.0);
  glLineWidth(6.0); // default is 1f
  glBegin(GL_LINES);
  glVertex3f(cx, 2., cy);
  glVertex3f(dirx + cx, 2., diry + cy);
  glEnd();
}

void drawGround(float x1, float y1, float x2, float y2) {
  glColor3f((GLfloat)0.0, (GLfloat)0.0, (GLfloat)0.0);
  glLineWidth(4.0); // default is 1f
  glBegin(GL_LINES);
  glVertex3f(x1, 2., y1);
  glVertex3f(x2, 2., y2);
  glEnd();
}

void write_config(Simulation *sim) {
  FILE *fp_out;
  std::string path = std::string(OUT_PATH) + std::string("config.txt");
  fp_out = fopen(path.c_str(), "w");
  if (fp_out != NULL) {
    fprintf(fp_out, "time_step %f\n", sim->time_step);
    for (int i = 0; i < sim->num_particles; i++) {
      fprintf(
          fp_out,
          "particle_id %d\tradius %f\tgoal %.3f %.3f\tcolor %.3f %.3f %.3f\n",
          i, sim->particles[i]->r, sim->particles[i]->goal.x,
          sim->particles[i]->goal.y, sim->particles[i]->color.x,
          sim->particles[i]->color.y, sim->particles[i]->color.z);
    }
  }
  fclose(fp_out);
}

void write_to_file(Simulation *sim) {
  FILE *fp_out;
  std::string frame = std::to_string(sim->step_no);
  std::string path = std::string(OUT_PATH) + frame + std::string(".txt");
  fp_out = fopen(path.c_str(), "w");
  if (fp_out != NULL) {
    for (int i = 0; i < sim->num_particles; i++) {
      fprintf(fp_out, "%d\t%.5f\t%.5f\n", i, sim->particles[i]->X.x,
              sim->particles[i]->X.y);
    }
  }
  fclose(fp_out);
}

void update(Simulation *sim) {
  write_to_file(sim);
  sim->do_time_step();
  // sim->do_time_step_force();
}

void draw_particles(Simulation *sim) {
  for (int i = 0; i < sim->num_particles; i++) {
    drawCircle(sim->particles[i]->X.x, sim->particles[i]->X.y, 1, 15,
               sim->particles[i]->color);
    /*
    drawDirection(sim->particles[i]->X.x,sim->particles[i]->X.y,
             sim->planner->velocity_buffer[i].x,sim->planner->velocity_buffer[i].y);
    */
  }

  for (int i = 0; i < sim->num_walls; i++) {
    Wall *w = sim->walls[i];
    drawGround(w->x0.x, w->x0.y, w->x1.x, w->x1.y);
  }

  drawGround(-1000.0, GROUND_HEIGHT, 1000.0, GROUND_HEIGHT);
  drawGround(-1000.0, GRID_UP_BOUND, 1000.0, GRID_UP_BOUND);
  drawGround(LEFT_BOUND_X, GROUND_HEIGHT, LEFT_BOUND_X, GROUND_HEIGHT - 1000);
  drawGround(RIGHT_BOUND_X, GROUND_HEIGHT, RIGHT_BOUND_X, GROUND_HEIGHT - 1000);
}

int render(Simulation *sim) {
  int display_w, display_h;
  double prev_time = 0.0;
  double elapsed = 0.0;
  double cur_time = 0.0;
  double lag = 0.0;
  BYTE pixels[3 * WIDTH * HEIGHT];
  glfwSetErrorCallback(error_callback);
  if (!glfwInit())
    return 1;
  GLFWwindow *window =
      glfwCreateWindow(WIDTH, HEIGHT, "ImGui Crowd Sim", NULL, NULL);
  glfwMakeContextCurrent(window);
  glEnable(GL_DEPTH_TEST); // Depth Testing
  glDepthFunc(GL_LEQUAL);
  glDisable(GL_CULL_FACE);
  glCullFace(GL_BACK);
  glfwGetWindowSize(window, &display_w, &display_h);
  glViewport(0, 0, display_w, display_h);
  float3 clear_color = make_float3(0.827f, 0.827f, 0.827f);

  prev_time = GetSeconds();
  // Main loop
  int i = 0;
  while (!glfwWindowShouldClose(window)) {
    glLoadIdentity();
    cur_time = GetSeconds();
    elapsed = cur_time - prev_time;
    prev_time = cur_time;
    lag += elapsed;

    while (lag >= MS_PER_UPDATE) {
      lag -= MS_PER_UPDATE;
      update(sim);
    }
    glClearColor(clear_color.x, clear_color.y, clear_color.z, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    draw_particles(sim);
    set_camera(display_w, display_h, 0.0 /*rotate_camera+=0.2*/);
    glfwSwapBuffers(window);
    glfwPollEvents();
    /*
            glReadPixels(0,0,display_w,display_h, GL_BGR, GL_UNSIGNED_BYTE,
    pixels); save_file(pixels,"hello",display_h,display_w); i++; if(i>10) break;
            */
  }
  glfwTerminate();
  return 0;
}

//-----------------------------------------
// rendering

float rand_interval(float min, float max) {
  return min + (max - min) * rand() / RAND_MAX;
}

float2 rand_float2(float minx, float maxx, float miny, float maxy) {
  return make_float2(rand_interval(minx, maxx), rand_interval(miny, maxy));
}

float deg_to_rad(float deg) {
  const float PI = 3.1415;
  while (deg >= 360.) {
    deg -= 360.;
  }
  while (deg <= 0.) {
    deg += 360.;
  }
  return PI / 180. * deg;
}


void init_from_file(Simulation *s)
{
  srand(time(NULL));

}

void dummy_init(Simulation *s) {
  srand(time(NULL));
  int i = 0;
  float rad = 3.2;
  float row_init = LEFT_BOUND_X + 3.8 * rad * 50 * 0.5;
  float height_init = GROUND_HEIGHT - 50;
  for (int i_ = 0; i_ < ROWS / 2; i_++) {
    for (int j_ = 0; j_ < COLS; j_++) {
      float y = height_init - rad * i_ + rand_interval(-0.4, 0.4);
      float x = row_init + j_ * rad + rand_interval(-0.4, 0.4);
      s->particles[i] =
          new Particle(make_float2(x, y), make_float2(0, 0.0), 1.0, 1.0, 0,
                       make_float3(0.0f, 0.0f, 1.0f),
                       make_float2(LEFT_BOUND_X + 2 * rad * 50 * 0.5, y));
      i++;
    }
  }

  rad = 3.1;
  height_init = GROUND_HEIGHT - 50 - 0.11;
  row_init = LEFT_BOUND_X + 1.4 * rad * 50 * 0.5;
  for (int i_ = 0; i_ < ROWS / 2; i_++) {
    for (int j_ = 0; j_ < COLS; j_++) {
      float y = height_init - rad * i_ + rand_interval(-0.4, 0.4);
      float x = row_init + j_ * rad + rand_interval(-0.4, 0.4);
      s->particles[i] =
          new Particle(make_float2(x, y), make_float2(0, 0.0), 1.0, 1.0, 0,
                       make_float3(1.0f, 0.0f, 0.0f),
                       make_float2(LEFT_BOUND_X + 8 * rad * 50 * 0.5 + x, y));
      i++;
    }
  }

  std::vector<particle_tuple *> friction_pairs = get_tuples(s->num_particles);
  int trig_len = 1 + (s->num_particles * (s->num_particles + 1) / 2);

  s->stability_upper_trig_arr =
      (Constraint **)malloc(sizeof(void *) * trig_len);
  s->collision_upper_trig_arr =
      (Constraint **)malloc(sizeof(void *) * trig_len);
  s->powerlaw_upper_trig_arr = (Constraint **)malloc(sizeof(void *) * trig_len);

  for (std::vector<particle_tuple *>::iterator it = friction_pairs.begin();
       it != friction_pairs.end(); ++it) {
    Stability_Constraint *stab =
        new Stability_Constraint(s, (*it)->i, (*it)->j);
    Friction_Constraint *fc = new Friction_Constraint(s, (*it)->i, (*it)->j);
    Powerlaw_Constraint *pl = new Powerlaw_Constraint(s, (*it)->i, (*it)->j);
    if ((*it)->i < (*it)->j) {
      s->collision_map[(*it)->i * s->num_particles + (*it)->j] = fc;
      int r = (*it)->i;
      int c = (*it)->j;
      int t_idx = (s->num_particles * r) + c - (r * (r + 1) * 0.5);
      s->collision_upper_trig_arr[t_idx] = fc;
      s->powerlaw_upper_trig_arr[t_idx] = pl;
      s->stability_upper_trig_arr[t_idx] = stab;
    }
  }

  // set up wall constraints
  s->num_walls = 2;
  s->walls = (Wall **)malloc(sizeof(void *) * s->num_walls);
  s->walls[0] =
      new Wall(make_float2(-170, GROUND_HEIGHT - 45),
               make_float2(150., GROUND_HEIGHT - 45), make_float2(0., 1.));
  s->walls[1] =
      new Wall(make_float2(-170, GROUND_HEIGHT - 75),
               make_float2(150., GROUND_HEIGHT - 75), make_float2(0., -1.));
  s->num_constraints =
      s->num_particles + s->num_particles * s->num_walls; // ground+walls
  s->constraints = (Constraint **)malloc(sizeof(void *) * s->num_constraints);
  int constraint_ctr = 0;


  for (int i = 0; i < s->num_particles; i++) {
    s->constraints[i] = new Ground_Constraint(s, i);
    constraint_ctr++;
  }
  for (int i = 0; i < s->num_particles; i++) {
    for (int j = 0; j < s->num_walls; j++) {
      s->constraints[constraint_ctr] = new Wall_Constraint(s, i, j);
      constraint_ctr++;
    }
  }

  for (int i = 0; i < s->num_particles; i++) {
    s->particles[i]->V_pref = V_PREF_ACCEL;

    float u;
    do {
      u = (float)rand() / (float)RAND_MAX;
    } while (u >= 1.0);
    s->particles[i]->V_pref +=
        sqrtf(-2.f * logf(1.f - u)) * 0.1f *
        cosf(2.f * _M_PI * (float)rand() / (float)RAND_MAX);

    s->planner->calc_pref_v_force(i);
    s->particles[i]->V.x = s->planner->velocity_buffer[i].x;
    s->particles[i]->V.y = s->planner->velocity_buffer[i].y;
  }
}

int main(int argc, char **argv) {
  // 0.03 sec - time step, 30 fr/sec
  int num_particles = ROWS * COLS;
  char *output = (char*)"blender.txt";
  Simulation sim(num_particles, 0, MS_PER_UPDATE, output);
  dummy_init(&sim);

  write_config(&sim);
  render(&sim);
  return 0;
}
