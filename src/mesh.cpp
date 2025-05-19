#include <map>

#include "mesh.h"
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

void Mesh::Bind() const
{
  if (m_VAO != 0)
  {
    glBindVertexArray(m_VAO);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    if (m_textures)
    {
      glEnableVertexAttribArray(3);
      glEnableVertexAttribArray(4);
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D_ARRAY, m_textures);
    }
    {
      GLenum err = glGetError();
			if (err != GL_NO_ERROR) { fprintf(stderr, "Error a! %d\n", err); }
    }
  }

  if (m_shaderSettings & Mesh::ShaderSettings::DrawBackfaces)
  {
    glDisable(GL_CULL_FACE);
  }
  else
  {
    glEnable(GL_CULL_FACE);
    glCullFace(GL_FRONT);
  }
  if (m_shaderSettings & Mesh::ShaderSettings::DrawWireframe)
  {
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  }
  else
  {
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  }
  if (m_shaderSettings & Mesh::ShaderSettings::ForceDrawOnTop)
  {
    glDisable(GL_DEPTH_TEST);
  }
  else
  {
    glEnable(GL_DEPTH_TEST);
  }
  if (m_shaderSettings & Mesh::ShaderSettings::DrawLinesAA)
  {
    glEnable(GL_LINE_SMOOTH);
  }
  else
  {
    glDisable(GL_LINE_SMOOTH);
  }
}

void Mesh::Unbind() const
{
  glBindVertexArray(0);
  glDisableVertexAttribArray(0);
  glDisableVertexAttribArray(1);
  glDisableVertexAttribArray(2);
  glDisableVertexAttribArray(3);
  glDisableVertexAttribArray(4);
  glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
  {
    GLenum err = glGetError();
		if (err != GL_NO_ERROR) { fprintf(stderr, "Error b! %d\n", err); }
  }
}

void Mesh::Draw() const
{
	if (m_VAO != 0)
  {
    glDrawArrays(GL_TRIANGLES, 0, m_dataBufSize / sizeof(float));
  }
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
void Mesh::UpdateMesh(const std::vector<float>& data, unsigned includedDataFlags, unsigned shadSettings, bool dataIsInterlaced)
{
  Dispose();

  includedDataFlags |= VBufDataType::VertexPos;

  m_dataBufSize = static_cast<unsigned>(data.size() * sizeof(float));
  m_includedData = includedDataFlags;
  m_shaderSettings = shadSettings;

  glGenVertexArrays(1, &m_VAO);
  glBindVertexArray(m_VAO);

  glGenBuffers(1, &m_VBO);
  glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
  glBufferData(GL_ARRAY_BUFFER, m_dataBufSize, data.data(), GL_STATIC_DRAW);

  int ultimateStrideSize = 0;
  for (size_t i = 0; i < sizeof(int) * 8; i++)
  {
    switch ((includedDataFlags) & (1 << i))
    {
    case VBufDataType::VertexPos: //dimension = 3
      ultimateStrideSize += 3;
      break;
    case VBufDataType::Barycentric: //dimension = 3
      ultimateStrideSize += 3;
      break;
    case VBufDataType::VColor: //dimension = 3
      ultimateStrideSize += 3;
      break;
    case VBufDataType::Normals: //dimension = 3
      ultimateStrideSize += 3;
      break;
    case VBufDataType::STUV: //dimension = 2
      ultimateStrideSize += 2;
      break;
    case VBufDataType::TexIndex: //dimension = 1
      ultimateStrideSize += 1;
      break;
    }
  }

  if (dataIsInterlaced)
  {
    //although this works, it's possible that what I'm trying to do here can be done much more simply via some opengl feature(s).
    for (int openglPositionCounter = 0, takenSize = 0, takenCount = 0; openglPositionCounter < 6 /*# of flags in VBufDataType*/; openglPositionCounter++)
    {
      switch (1 << openglPositionCounter)
      {
			case VBufDataType::VertexPos: //position
        if ((includedDataFlags & VBufDataType::VertexPos) != 0)
        {
          constexpr int dim = 3;
          glVertexAttribPointer(takenCount, dim, GL_FLOAT, GL_FALSE, ultimateStrideSize * sizeof(float), (void*)(takenSize * sizeof(float)));
          glEnableVertexAttribArray(takenCount);
          takenCount++;
          takenSize += dim;
        }
        break;
			case VBufDataType::Barycentric: //barycentric
        if ((includedDataFlags & VBufDataType::Barycentric) != 0)
        {
          constexpr int dim = 3;
          glVertexAttribPointer(takenCount, dim, GL_FLOAT, GL_FALSE, ultimateStrideSize * sizeof(float), (void*)(takenSize * sizeof(float)));
          glEnableVertexAttribArray(takenCount);
          takenCount++;
          takenSize += dim;
        }
        break;
      case VBufDataType::VColor:
        if ((includedDataFlags & VBufDataType::VColor) != 0)
        {
          constexpr int dim = 3;
          glVertexAttribPointer(takenCount, dim, GL_FLOAT, GL_FALSE, ultimateStrideSize * sizeof(float), (void*)(takenSize * sizeof(float)));
          glEnableVertexAttribArray(takenCount);
          takenCount++;
          takenSize += dim;
        }
        break;
      case VBufDataType::Normals:
        if ((includedDataFlags & VBufDataType::Normals) != 0)
        {
          constexpr int dim = 3;
          glVertexAttribPointer(takenCount, dim, GL_FLOAT, GL_FALSE, ultimateStrideSize * sizeof(float), (void*)(takenSize * sizeof(float)));
          glEnableVertexAttribArray(takenCount);
          takenCount++;
          takenSize += dim;
        }
        break;
      case VBufDataType::STUV:
        if ((includedDataFlags & VBufDataType::STUV) != 0)
        {
          constexpr int dim = 2;
          glVertexAttribPointer(takenCount, dim, GL_FLOAT, GL_FALSE, ultimateStrideSize * sizeof(float), (void*)(takenSize * sizeof(float)));
          glEnableVertexAttribArray(takenCount);
          takenCount++;
          takenSize += dim;
        }
        break;
      case VBufDataType::TexIndex:
        if ((includedDataFlags & VBufDataType::TexIndex) != 0)
        {
          //this only works because sizeof(float) == sizeof(int)
          constexpr int dim = 1;
          glVertexAttribIPointer(takenCount, dim, GL_INT, ultimateStrideSize * sizeof(int), (void*)(takenSize * sizeof(int)));
          glEnableVertexAttribArray(takenCount);
          takenCount++;
          takenSize += dim;
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

int Mesh::GetDatas() const
{
  return m_includedData;
}

int Mesh::GetShaderSettings() const
{
  return m_shaderSettings;
}

void Mesh::SetShaderSettings(unsigned shadSettings)
{
  m_shaderSettings = shadSettings;
}

void Mesh::SetTextureStore(const std::map<int, std::filesystem::path>& texturePaths)
{
  if (m_textures) { glDeleteTextures(1, &m_textures); m_textures = 0; }

  glGenTextures(1, &m_textures);
  glBindTexture(GL_TEXTURE_2D_ARRAY, m_textures);

  glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA8, 256, 256, static_cast<GLsizei>(texturePaths.size()), 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

  for (int texIndex = 0; texIndex < static_cast<int>(texturePaths.size()); texIndex++)
  {
    const std::filesystem::path& path = texturePaths.at(texIndex);
    int w, h, channels;
    stbi_uc* originalData;
    originalData = stbi_load(path.generic_string().c_str(), &w, &h, &channels, 0);

    constexpr int ww = 256, hh = 256;
    static unsigned char finalData[ww * hh * 4];

    if (originalData == nullptr)
    {
      printf("Failed to load texture: \"%s\", defaulting to 50%% grey.\n", path.generic_string().c_str());

      memset(finalData, 0x7F, ww * hh * 4);
      for (size_t i = 0; i < ww * hh * 4; i += 4)
        finalData[i + 3] = 0xFF;

      glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, static_cast<GLint>(texIndex), 256, 256, 1, GL_RGBA, GL_UNSIGNED_BYTE, (const void*)finalData);
    }
    else
    {
      memset(finalData, 0xFF, ww * hh * 4);

      stbir_resize(originalData, w, h, channels * w, finalData, ww, hh, 4 * ww, stbir_pixel_layout::STBIR_RGBA, stbir_datatype::STBIR_TYPE_UINT8, stbir_edge::STBIR_EDGE_CLAMP, stbir_filter::STBIR_FILTER_POINT_SAMPLE);

      //stbir_resize_uint8_srgb(originalData, w, h, channels * w, finalData, ww, hh, 4 * ww, stbir_pixel_layout::STBIR_RGBA);

      glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, static_cast<GLint>(texIndex), 256, 256, 1, GL_RGBA, GL_UNSIGNED_BYTE, (const void*)finalData);
    }
  }
}

GLuint Mesh::GetTextureStore()
{
  return m_textures;
}

void Mesh::Dispose()
{
	if (m_VAO != 0) { glDeleteVertexArrays(1, &m_VAO); m_VAO = 0; }
	if (m_VBO != 0) { glDeleteBuffers(1, &m_VBO); m_VBO = 0; }
  m_dataBufSize = 0;
}