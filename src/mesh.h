#pragma once

#include "globalimguiglglfw.h"
#include "glm.hpp"
#include "gtc/matrix_transform.hpp"
#include "gtc/type_ptr.hpp"

class Mesh {
public:
  enum ShaderSettings //note: updating this also requires updating all shader source.
  {
    None = 0,
    DrawWireframe = 1,
    DrawBackfaces = 4,
  };
  enum VBufDataType
  { //all are assumed to have vertex data (otherwise, what the hell are we drawing?)
    VertexPos = 1, //implicitly always on
    Barycentric = 2,
    VColor = 4,
    Normals = 8,
    STUV_1 = 16, //tex coords
  };
private:
  GLuint VAO = 0, VBO = 0;
  int dataBufSize = 0;
  int shaderSettings;
  int includedData;
public:

  //don't call these three directly. Use the Model class.
  void Bind();
  void Unbind();
  void Draw();
  int GetDatas();
  int GetShaderSettings();
  void SetShaderSettings(int shadSettings);
  void UpdateMesh(float data[], int dataBufSize, int includedDataFlags, int shadSettings, bool dataIsInterlaced = true);
  Mesh(float data[], int dataBufSize, int includedDataFlags, int shadSettings, bool dataIsInterlaced = true);
  Mesh();
  void Dispose();
};