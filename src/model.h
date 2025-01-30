#pragma once

#include "glm.hpp"
#include "common.hpp"
#include "gtc/matrix_transform.hpp"
#include "gtc/type_ptr.hpp"
#include "gtc/quaternion.hpp"
#include "mesh.h"

class Model
{
private:
  Mesh* mesh;
public:
  glm::vec3 position;
  glm::vec3 scale;
  glm::quat rotation;
  Model(Mesh* mesh, glm::vec3 position = glm::vec3(0.f, 0.f, 0.f), glm::vec3 scale = glm::vec3(1.f, 1.f, 1.f), glm::quat rotation = glm::quat(1.f, 0.f, 0.f, 0.f));
  Model();
  glm::mat4 CalculateModelMatrix();
  Mesh* GetMesh();
  void SetMesh(Mesh* newMesh = nullptr);
  void Draw();
};