#include "mesh.h"

void Mesh::Bind()
{
  if (VAO != 0)
  {
    glBindVertexArray(VAO);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glEnableVertexAttribArray(3);
    {
      GLenum err = glGetError();
      if (err != GL_NO_ERROR)
        fprintf(stderr, "Error a! %d\n", err);
    }
  }

  if (Mesh::shaderSettings & Mesh::ShaderSettings::DrawBackfaces)
  {
    glDisable(GL_CULL_FACE);
  }
  else
  {
    glEnable(GL_CULL_FACE);
    glCullFace(GL_FRONT);
  }
  if (Mesh::shaderSettings & Mesh::ShaderSettings::DrawWireframe)
  {
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  }
  else
  {
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  }
}

void Mesh::Unbind()
{
  glBindVertexArray(0);
  glDisableVertexAttribArray(0);
  glDisableVertexAttribArray(1);
  glDisableVertexAttribArray(2);
  glDisableVertexAttribArray(3);
  {
    GLenum err = glGetError();
    if (err != GL_NO_ERROR)
      fprintf(stderr, "Error b! %d\n", err);
  }
}

void Mesh::Draw()
{
  GLuint currentProgram;
  glGetIntegerv(GL_CURRENT_PROGRAM, (GLint*)&currentProgram);
  if (VAO != 0)
    glDrawArrays(GL_TRIANGLES, 0, dataBufSize / sizeof(float));
}

/// <summary>
/// When passing data[], any present data according to the "includedDataFlags" is expected to be in this order:
/// 
/// vertex/position data (always assumed to be present).
/// barycentric (1, 0, 0), (0, 1, 0), (0, 0, 1).
/// normal
/// vcolor
/// stuv_1
/// </summary>
void Mesh::UpdateMesh(float data[], int dataBufSize, int includedDataFlags, int shadSettings, bool dataIsInterlaced)
{
  Dispose();

  includedDataFlags |= VBufDataType::VertexPos;

  this->dataBufSize = dataBufSize;
  this->includedData = includedDataFlags;
  this->shaderSettings = shadSettings;

  glGenVertexArrays(1, &VAO);
  glBindVertexArray(VAO);

  glGenBuffers(1, &VBO);
  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, dataBufSize, data, GL_STATIC_DRAW);

  int ultimateStrideSize = 0;
  //count set data flags
  for (size_t i = 0; i < 32; i++)
  {
    switch (((int)includedDataFlags) & (1 << i))
    {
    case (int)VBufDataType::VertexPos: //dimension = 3
      ultimateStrideSize += 3;
      break;
    case (int)VBufDataType::Barycentric: //dimension = 3
      ultimateStrideSize += 3;
      break;
    case (int)VBufDataType::VColor: //dimension = 3
      ultimateStrideSize += 3;
      break;
    case (int)VBufDataType::Normals: //dimension = 3
      ultimateStrideSize += 3;
      break;
    case (int)VBufDataType::STUV_1: //undecided 2/4 idk probably 2
      fprintf(stderr, "Unimplemented VBufDataType::STUV_1 in Mesh::UpdateMesh()");
      throw 0;
      break;
    }
  }

  if (dataIsInterlaced)
  {
    //although this works, it's possible that what I'm trying to do here can be done much more simply via some opengl feature(s).
    for (int openglPositionCounter = 0, takenSize = 0, takenCount = 0; openglPositionCounter < 5 /*# of flags in VBufDataType*/; openglPositionCounter++)
    {
      switch (1 << openglPositionCounter)
      {
      case 1: //position
        {
          constexpr int dim = 3;
          glVertexAttribPointer(takenCount, dim, GL_FLOAT, GL_FALSE, ultimateStrideSize * sizeof(float), (void*)(takenSize * sizeof(float)));
          glEnableVertexAttribArray(takenCount);
          takenCount++;
          takenSize += dim;
        }
        break;
      case 2: //barycentric
        {
        constexpr int dim = 3;
        glVertexAttribPointer(takenCount, dim, GL_FLOAT, GL_FALSE, ultimateStrideSize * sizeof(float), (void*)(takenSize * sizeof(float)));
        glEnableVertexAttribArray(takenCount);
        takenCount++;
        takenSize += dim;
        }
        break;
      case VBufDataType::VColor:
        if (includedDataFlags & VBufDataType::VColor)
        {
          constexpr int dim = 3;
          glVertexAttribPointer(takenCount, dim, GL_FLOAT, GL_FALSE, ultimateStrideSize * sizeof(float), (void*)(takenSize * sizeof(float)));
          glEnableVertexAttribArray(takenCount);
          takenCount++;
          takenSize += dim;
        }
        break;
      case VBufDataType::Normals:
        if (includedDataFlags & VBufDataType::Normals)
        {
          constexpr int dim = 3;
          glVertexAttribPointer(takenCount, dim, GL_FLOAT, GL_FALSE, ultimateStrideSize * sizeof(float), (void*)(takenSize * sizeof(float)));
          glEnableVertexAttribArray(takenCount);
          takenCount++;
          takenSize += dim;
        }
        break;
      case VBufDataType::STUV_1:
        if (includedDataFlags & VBufDataType::STUV_1)
        {
          fprintf(stderr, "Unimplemented VBufDataType::STUV_1 in Mesh::UpdateMesh()");
          throw 0;
        }
        break;
      }
    }
  }
  else
  {
    fprintf(stderr, "Unimplemented dataIsInterlaced=false in Mesh::UpdateMesh()");
    throw 0;
  }

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);
}

int Mesh::GetDatas()
{
  return this->includedData;
}

int Mesh::GetShaderSettings()
{
  return this->shaderSettings;
}

void Mesh::SetShaderSettings(int shadSettings)
{
  this->shaderSettings = shadSettings;
}

Mesh::Mesh(float data[], int dataBufSize, int includedDataFlags, int shadSettings, bool dataIsInterlaced)
{
  UpdateMesh(data, dataBufSize, includedDataFlags, shadSettings, dataIsInterlaced);
}

Mesh::Mesh()
{

}

void Mesh::Dispose()
{
  if (VAO != 0)
    glDeleteVertexArrays(1, &VAO);
  VAO = 0;
  if (VBO != 0)
    glDeleteBuffers(1, &VBO);
  VBO = 0;
  dataBufSize = 0;
}