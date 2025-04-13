#include "texture.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

static constexpr size_t MIN_CLUT_WIDTH = 16;
static constexpr size_t TEXPAGE_WIDTH = 64;
static constexpr size_t TEXPAGE_HEIGHT = 256;
static constexpr size_t VRAM_WIDTH = 512;
static constexpr size_t VRAM_HEIGHT = 512;
static constexpr size_t RESERVED_TEXPAGES[] = {6, 7};

static size_t GetTexPage(size_t x, size_t y)
{
	return (x / TEXPAGE_WIDTH) + ((VRAM_WIDTH / TEXPAGE_WIDTH) * (y / TEXPAGE_HEIGHT));
}

Texture::Texture(const std::filesystem::path& path)
{
	m_path = path;
	m_blendMode = PSX::BlendMode::HALF_TRANSPARENT;
	CreateTexture();
}

void Texture::UpdateTexture(const std::filesystem::path& path)
{
	uint16_t blendMode = m_blendMode;
	ClearTexture();
	m_path = path;
	m_blendMode = blendMode;
	CreateTexture();
}

Texture::BPP Texture::GetBPP() const
{
	size_t colorCount = m_clut.size();
	if (colorCount <= 16) { return Texture::BPP::BPP_4; }
	else if (colorCount <= 256) { return Texture::BPP::BPP_8; }
	return Texture::BPP::BPP_16;
}

int Texture::GetWidth() const
{
	return m_width;
}

int Texture::GetVRAMWidth() const
{
	int ret = m_width;
	Texture::BPP bpp = GetBPP();
	if (bpp == Texture::BPP::BPP_4) { ret /= 4; }
	else if (bpp == Texture::BPP::BPP_8) { ret /= 2; }
	return ret;
}

int Texture::GetHeight() const
{
	return m_height;
}

uint16_t Texture::GetBlendMode() const
{
  return m_blendMode;
}

const std::filesystem::path& Texture::GetPath() const
{
	return m_path;
}

bool Texture::Empty() const
{
	return m_width == 0;
}

const std::vector<uint16_t>& Texture::GetImage() const
{
	return m_image;
}

const std::vector<uint16_t>& Texture::GetClut() const
{
	return m_clut;
}

size_t Texture::GetImageX() const
{
	return m_imageX - 512;
}

size_t Texture::GetImageY() const
{
	return m_imageY;
}

size_t Texture::GetCLUTX() const
{
	return m_clutX - 512;
}

size_t Texture::GetCLUTY() const
{
	return m_clutY;
}

void Texture::SetImageCoords(size_t x, size_t y)
{
	m_imageX = x + 512;
	m_imageY = y;
}

void Texture::SetCLUTCoords(size_t x, size_t y)
{
	m_clutX = x + 512;
	m_clutY = y;
}

void Texture::SetBlendMode(size_t mode)
{
	m_blendMode = static_cast<uint16_t>(mode);
}

PSX::TextureLayout Texture::Serialize(const QuadUV& uvs, bool lowLOD)
{
	PSX::TextureLayout layout = {};
	if (Empty()) { return layout; }

	layout.texPage.blendMode = m_blendMode;
	size_t bppMultiplier = 1;
	Texture::BPP bpp = GetBPP();
	switch (bpp)
	{
	case Texture::BPP::BPP_4:
		layout.texPage.texpageColors = 0;
		bppMultiplier = 4;
		break;
	case Texture::BPP::BPP_8:
		layout.texPage.texpageColors = 1;
		bppMultiplier = 2;
		break;
	case Texture::BPP::BPP_16:
		layout.texPage.texpageColors = 2;
		break;
	}
	layout.texPage.x = static_cast<uint16_t>(m_imageX / TEXPAGE_WIDTH);
	layout.texPage.y = static_cast<uint16_t>(m_imageY / TEXPAGE_HEIGHT);

	layout.clut.x = static_cast<uint16_t>(m_clutX / MIN_CLUT_WIDTH);
	layout.clut.y = static_cast<uint16_t>(m_clutY);

	uint8_t x = static_cast<uint8_t>((m_imageX % TEXPAGE_WIDTH) * bppMultiplier);
	uint8_t y = static_cast<uint8_t>(m_imageY % TEXPAGE_HEIGHT);
	float width = static_cast<float>(GetWidth() - 1);
	float height = static_cast<float>(GetHeight() - 1);
	layout.u0 = x + static_cast<uint8_t>(std::round(uvs[0].x * width));	layout.v0 = y + static_cast<uint8_t>(std::round(uvs[0].y * height));
	layout.u1 = x + static_cast<uint8_t>(std::round(uvs[1].x * width));	layout.v1 = y + static_cast<uint8_t>(std::round(uvs[1].y * height));
	layout.u2 = x + static_cast<uint8_t>(std::round(uvs[2].x * width));	layout.v2 = y + static_cast<uint8_t>(std::round(uvs[2].y * height));
	layout.u3 = x + static_cast<uint8_t>(std::round(uvs[3].x * width));	layout.v3 = y + static_cast<uint8_t>(std::round(uvs[3].y * height));

	if (!lowLOD)
	{
		auto FixOffByOne = [](const uint8_t& n0, uint8_t& n1, float expected)
			{
				n1 -= static_cast<uint8_t>(static_cast<int>(n1 - n0) - static_cast<int>(std::trunc(expected)));
			};

		FixOffByOne(layout.u0, layout.u1, (uvs[1].x * width) - (uvs[0].x * width));
		FixOffByOne(layout.u2, layout.u3, (uvs[3].x * width) - (uvs[2].x * width));
		FixOffByOne(layout.v0, layout.v2, (uvs[2].y * height) - (uvs[0].y * height));
		FixOffByOne(layout.v1, layout.v3, (uvs[3].y * height) - (uvs[1].y * height));
	}

	return layout;
}

bool Texture::CompareEquivalency(const Texture& tex)
{
	Texture::BPP bpp = GetBPP();
	if ((bpp == Texture::BPP::BPP_16) || (GetWidth() != tex.GetWidth()) || (GetHeight() != tex.GetHeight()) || (bpp != tex.GetBPP())) { return false; }
	for (const Shape& aShape : m_shapes)
	{
		bool foundShape = false;
		size_t idx = *aShape.begin();
		for (const Shape& bShape : tex.m_shapes)
		{
			if (!bShape.contains(idx)) { continue; }

			foundShape = true;
			for (size_t bIdx : bShape)
			{
				if (!aShape.contains(bIdx)) { return false; }
			}
			break;
		}
		if (!foundShape) { return false; }
	}
	return true;
}

void Texture::CopyVRAMAttributes(const Texture& tex)
{
	SetImageCoords(tex.GetImageX(), tex.GetImageY());
	SetCLUTCoords(tex.GetCLUTX(), tex.GetCLUTY());
}

bool Texture::operator==(const Texture& tex) const
{
	return (GetWidth() == tex.GetWidth()) && (GetHeight() == tex.GetHeight()) && (GetBPP() == tex.GetBPP()) && m_clut == tex.m_clut && m_image == tex.m_image;
}

void Texture::FillShapes(const std::vector<size_t>& colorIndexes)
{
	Texture::BPP bpp = GetBPP();
	if (bpp == Texture::BPP::BPP_16) { return; }

	for (size_t i = 0; i < m_clut.size(); i++)
	{
		Shape shape;
		for (size_t j = 0; j < colorIndexes.size(); j++)
		{
			if (colorIndexes[j] == i) { shape.insert(j); }
		}
		m_shapes.push_back(shape);
	}
}

void Texture::ClearTexture()
{
	m_blendMode = 0;
	m_width = m_height = 0;
	m_imageX = m_imageY = 0;
	m_clutX = m_clutY = 0;
	m_image.clear(); m_clut.clear();
	m_path.clear(); m_shapes.clear();
}

void Texture::CreateTexture()
{
	int channels;
	stbi_uc* image = stbi_load(m_path.string().c_str(), &m_width, &m_height, &channels, 0);
	bool alphaImage = channels == 4;
	std::vector<size_t> colorIndexes;
	for (int i = 0; i < m_width * m_height; i++)
	{
		int px = i * channels;
		uint16_t color = alphaImage ? ConvertColor(image[px + 0], image[px + 1], image[px + 2], image[px + 3]) :
																	ConvertColor(image[px + 0], image[px + 1], image[px + 2], 255);
		bool foundColor = false;
		size_t clutIndex = m_clut.size();
		for (size_t j = 0; j < m_clut.size(); j++)
		{
			if (color == m_clut[j]) { clutIndex = j; foundColor = true; break; }
		}
		if (!foundColor) { m_clut.push_back(color); }
		colorIndexes.push_back(clutIndex);
	}
	Texture::BPP bpp = GetBPP();
	if (bpp == Texture::BPP::BPP_4) { ConvertPixels(colorIndexes, 4); }
	else if (bpp == Texture::BPP::BPP_8) { ConvertPixels(colorIndexes, 2); }
	else { m_image = m_clut; }
	FillShapes(colorIndexes);
	stbi_image_free(image);
}

uint16_t Texture::ConvertColor(unsigned char r, unsigned char g, unsigned char b, unsigned char a)
{
	if (a == 255)
	{
		a = 0;
		if (r == 0 && g == 0 && b == 0) { b = 8; }
	}
	else if (a == 0) { r = g = b = 0; }
	else { a = 1; }
	uint16_t color = a << 5;
	color |= (((b * 249) + 1014) >> 11) & 0x1F;
	color <<= 5;
	color |= (((g * 249) + 1014) >> 11) & 0x1F;
	color <<= 5;
	color |= (((r * 249) + 1014) >> 11) & 0x1F;
	return color;
}

void Texture::ConvertPixels(const std::vector<size_t>& colorIndexes, unsigned indexesPerPixel)
{
	uint16_t px = 0;
	int relWidth = 0;
	unsigned colorsAdded = 0;
	unsigned shifter = (sizeof(uint16_t) * 8) / indexesPerPixel;
	for (size_t index : colorIndexes)
	{
		if (relWidth == GetWidth()) { relWidth = 0; m_image.push_back(px); px = 0; colorsAdded = 0; }
		else if (colorsAdded == indexesPerPixel) { m_image.push_back(px); px = 0; colorsAdded = 0; }
		px |= static_cast<uint16_t>(index << (shifter * colorsAdded));
		colorsAdded++;
		relWidth++;
	}
	m_image.push_back(px);
}

static constexpr size_t GetVRAMLocation(size_t x, size_t y)
{
	return x + (y * VRAM_WIDTH);
}

static void BufferToVRM(std::vector<uint16_t>& vram, std::vector<bool>& vramUsed, const std::vector<uint16_t>& buffer, size_t x, size_t y, size_t width)
{
	size_t relWidth = 0;
	size_t relHeight = 0;
	for (uint16_t px : buffer)
	{
		if (relWidth == width) { relWidth = 0; relHeight++; }
		size_t coord = GetVRAMLocation(x + relWidth, y + relHeight);
		vram[coord] = px;
		vramUsed[coord] = true;
		relWidth++;
	}
}

static bool TestRect(std::vector<bool>& vramUsed, size_t x, size_t y, size_t width, size_t height, bool clut)
{
	if (((x + width) >= VRAM_WIDTH) || (y + height) >= VRAM_HEIGHT) { return false; }

	size_t texPage = GetTexPage(x, y);
	size_t texPageLastCorner = GetTexPage(x + width - 1, y + height - 1);
	if (texPage == RESERVED_TEXPAGES[0] || texPage == RESERVED_TEXPAGES[1]) { return false; }
	if (texPageLastCorner == RESERVED_TEXPAGES[0] || texPageLastCorner == RESERVED_TEXPAGES[1]) { return false; }
	if (!clut && texPage != texPageLastCorner) { return false; }

	size_t relWidth = 0;
	size_t relHeight = 0;
	for (size_t i = 0; i < width * height; i++)
	{
		if (relWidth == width) { relWidth = 0; relHeight++; }
		if (vramUsed[GetVRAMLocation(x + relWidth, y + relHeight)]) { return false; }
		relWidth++;
	}
	return true;
}

static bool FindAvailableSpace(std::vector<bool>& vramUsed, size_t width, size_t height, size_t& retX, size_t& retY, bool clut)
{
	size_t x = 0;
	size_t y = 0;
	while (GetVRAMLocation(x, y) < vramUsed.size())
	{
		if (x == VRAM_WIDTH) { x = 0; y++; }
		if (TestRect(vramUsed, x, y, width, height, clut))
		{
			retX = x;
			retY = y;
			return true;
		}
		if (clut) { x += 16; }
		else { x++; }
		if (((x + width) >= VRAM_WIDTH) && (y + height) >= VRAM_HEIGHT) { break; }
	}
	return false;
}

std::vector<uint8_t> PackVRM(std::vector<Texture*>& textures)
{
	bool empty = true;
	std::vector<bool> vramUsed(VRAM_WIDTH * VRAM_HEIGHT, false);
	std::vector<uint16_t> vram(VRAM_WIDTH * VRAM_HEIGHT, 0);
	std::vector<Texture*> cachedTextures;

	for (Texture* texture : textures)
	{
		if (texture->Empty()) { continue; }

		bool foundEquivalent = false;
		for (Texture* cachedTexture : cachedTextures)
		{
			if (texture->CompareEquivalency(*cachedTexture))
			{
				texture->SetImageCoords(cachedTexture->GetImageX(), cachedTexture->GetImageY());
				foundEquivalent = true;
				break;
			}
		}

		if (foundEquivalent) { continue; }

		size_t x, y;
		if (!FindAvailableSpace(vramUsed, texture->GetVRAMWidth(), texture->GetHeight(), x, y, false))
		{
			return std::vector<uint8_t>();
		}
		empty = false;
		texture->SetImageCoords(x, y);
		BufferToVRM(vram, vramUsed, texture->GetImage(), x, y, texture->GetVRAMWidth());
		cachedTextures.push_back(texture);
	}

	if (empty) { return std::vector<uint8_t>(); }

	for (Texture* texture : textures)
	{
		if (texture->Empty()) { continue; }

		Texture::BPP bpp = texture->GetBPP();
		if (bpp == Texture::BPP::BPP_16) { continue; }

		size_t x, y;
		const std::vector<uint16_t>& clut = texture->GetClut();

		if (!FindAvailableSpace(vramUsed, clut.size(), 1, x, y, true))
		{
			return std::vector<uint8_t>();
		}
		texture->SetCLUTCoords(x, y);
		BufferToVRM(vram, vramUsed, clut, x, y, clut.size());
	}

	constexpr size_t vrmSize = 0x70038;
	std::vector<uint8_t> vrm(vrmSize);
	uint8_t* pVrm = vrm.data();

	constexpr uint32_t vrmMagic = 0x20;
	memcpy(pVrm, &vrmMagic, sizeof(uint32_t)); pVrm += sizeof(uint32_t);

	constexpr size_t texpageSize = TEXPAGE_WIDTH * TEXPAGE_HEIGHT * sizeof(uint16_t);
	constexpr size_t buffer_1_size = texpageSize * 6;
	constexpr size_t buffer_2_size = texpageSize * 8;

	PSX::VRMHeader header_1 = {};
	header_1.size = static_cast<uint32_t>(buffer_1_size + 0x14);
	header_1.magic = 0x10;
	header_1.flags = 0x2;
	header_1.len = static_cast<uint32_t>(buffer_1_size + 0xC);
	header_1.x = 512;
	header_1.y = 0;
	header_1.width = static_cast<uint16_t>(TEXPAGE_WIDTH * 6);
	header_1.height = static_cast<uint16_t>(TEXPAGE_HEIGHT);
	memcpy(pVrm, &header_1, sizeof(PSX::VRMHeader)); pVrm += sizeof(PSX::VRMHeader);

	for (size_t i = 0; i < TEXPAGE_HEIGHT; i++)
	{
		size_t buffer_1_Location = GetVRAMLocation(0, i);
		size_t cpySize = 6 * TEXPAGE_WIDTH * sizeof(uint16_t);
		memcpy(pVrm, &vram[buffer_1_Location], cpySize); pVrm += cpySize;
	}

	PSX::VRMHeader header_2 = {};
	header_2.size = static_cast<uint32_t>(buffer_2_size + 0x14);
	header_2.magic = 0x10;
	header_2.flags = 0x2;
	header_2.len = static_cast<uint32_t>(buffer_2_size + 0xC);
	header_2.x = 512;
	header_2.y = static_cast<uint16_t>(TEXPAGE_HEIGHT);
	header_2.width = static_cast<uint16_t>(TEXPAGE_WIDTH * 8);
	header_2.height = static_cast<uint16_t>(TEXPAGE_HEIGHT);
	memcpy(pVrm, &header_2, sizeof(PSX::VRMHeader)); pVrm += sizeof(PSX::VRMHeader);

	constexpr size_t buffer_2_Location = GetVRAMLocation(0, TEXPAGE_HEIGHT);
	memcpy(pVrm, &vram[buffer_2_Location], buffer_2_size); pVrm += buffer_2_size;
	return vrm;
}