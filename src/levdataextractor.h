#pragma once

#include <filesystem>
#include <vector>
#include <cstdint>
#include <string>

#include "psx_types.h"

namespace SH
{
	// Result of dry-parsing a model's command list.
	struct CommandListScan
	{
		bool valid = false;
		size_t numCommands = 0;        // excluding the 0xFFFFFFFF terminator
		size_t numVerts = 0;           // == the renderer's final ctx->vertexIndex
		uint32_t colorCount = 0;       // word 0 -- authoritative color-array length
		uint32_t maxTexCoordIndex = 0; // from non-color-only commands only
		uint32_t maxColorCoordIndex = 0;
		size_t byteSize = 0;           // 4 + (numCommands + 1) * 4
		const char* rejectReason = nullptr;
	};

	// Validated byte extents of one ModelAnim and its vertex data.
	struct AnimExtent
	{
		uint32_t offAnim = 0;
		uint32_t offDeltaArray = 0; // 0 if uncompressed
		size_t numStoredFrames = 0;
		size_t frameSize = 0;
		size_t animBlockBytes = 0;  // 0x18 + numStoredFrames * frameSize
		size_t deltaArrayBytes = 0; // numVerts * 4, or 0
		size_t payloadBytes = 0;    // cross-check only; sizing uses frameSize
		size_t vertexOffset = 0;    // uniform across this animation's frames; usually 0x1C
		std::string rejectReason;
	};

	// A byte range claimed from the source LEV, used to self-check that no two blocks
	// overlap (which would mean a computed size is too large).
	struct LevExtent
	{
		uint32_t start = 0;
		uint32_t end = 0;
		std::string label;
	};
}

class LevDataExtractor
{
public:
	LevDataExtractor(const std::filesystem::path& levPath, const std::filesystem::path& vrmPath);

	void ExtractModels(void);
	const std::string& GetLog() const { return m_log; }

	// Texture extraction helpers (public for use by free functions)
	static constexpr size_t VRAM_WIDTH = 1024;
	static constexpr size_t VRAM_HEIGHT = 512;

	struct VramBuffer {
		std::vector<uint16_t> data; // 1024x512 ushorts
		VramBuffer() : data(VRAM_WIDTH * VRAM_HEIGHT, 0) {}
	};

private:
	void Log(const char* format, ...);

	// VRAM operations
	void ParseVrmIntoVram(VramBuffer& vram);

	// Bounds-checked access to LEV data. Offsets are from start of file + 4 (see Deref).
	bool InLevBounds(uint32_t offset, size_t size) const;

	// Dry-parses a command list, mirroring RenderBucket_DrawFunc_Normal. Returns false
	// (and fills out.rejectReason) if the list is out of bounds or unterminated.
	bool ScanCommandList(uint32_t offCommandList, SH::CommandListScan& out);

	// Exact byte length of a compressed vertex payload, derived purely from the delta
	// array's per-vertex bit widths. outSafeReadBytes is the (larger) extent the game
	// may actually dereference.
	bool ComputeCompressedPayload(uint32_t offDeltaArray, size_t numVerts,
	                              size_t& outPayloadBytes, size_t& outSafeReadBytes);

	// Validates one ModelAnim and computes its byte extents. numVerts comes from the
	// owning ModelHeader's command list scan.
	bool ValidateAnim(uint32_t offAnim, size_t numVerts, SH::AnimExtent& out);

	// Self-check: no two blocks claimed from the LEV may partially overlap.
	bool ValidateExtents(std::vector<SH::LevExtent>& extents, const char* modelName);

	std::filesystem::path m_levPath;
	std::filesystem::path m_vrmPath;
	std::vector<uint8_t> m_levData;
	std::vector<uint8_t> m_vrmData;
	std::string m_log;
};

namespace SH
{
	struct WriteableObject
	{
		size_t size;
		void* data;
	};

	struct CtrModel
	{
		uint32_t modelOffset;
		uint32_t modelPatchTableOffset;
		uint32_t textureDataOffset; // Points to TextureSectionHeader, 0 if no textures
	};

	struct TextureSectionHeader
	{
		uint32_t numTextures;
		// Followed by: uint32_t textureOffsets[numTextures] (offsets relative to file start)
		// Followed by: TextureData for each texture (at the offsets specified above)
	};

	struct TextureDataHeader
	{
		uint16_t width;
		uint16_t height;
		uint8_t bpp;        // 0=4bit, 1=8bit, 2=16bit
		uint8_t blendMode;
		uint8_t originU;    // minU of combined texture in original UV space
		uint8_t originV;    // minV of combined texture in original UV space
		// Original VRAM coordinates (for matching TextureLayouts to textures at import)
		uint8_t origPageX;  // Original texpage X (0-15)
		uint8_t origPageY;  // Original texpage Y (0-1)
		uint8_t origPalX;   // Original CLUT X / 16 (0-63)
		uint8_t origPalY_lo; // Original CLUT Y low byte
		uint8_t origPalY_hi; // Original CLUT Y high byte (Y is 0-511)
		uint8_t padding;    // Align to 16 bytes total
		// Followed by:
		// - Pixel data in raw PSX format:
		//   4-bit: ceil(width/2) * height bytes (2 pixels per byte)
		//   8-bit: width * height bytes
		//   16-bit: width * height * 2 bytes
		// - Palette data (raw PSX 16-bit colors):
		//   4-bit: 16 * sizeof(uint16_t) = 32 bytes
		//   8-bit: 256 * sizeof(uint16_t) = 512 bytes
		//   16-bit: no palette
	};

	// Groups multiple TextureLayouts that share the same texpage+palette into a single combined texture for extraction
	struct TextureExtractionGroup
	{
		std::vector<size_t> layoutIndices;  // Indices of TextureLayouts in this group
		int minU = 255;
		int minV = 255;
		int maxU = 0;
		int maxV = 0;
		int pageX = 0;
		int pageY = 0;
		int palX = 0;
		int palY = 0;
		int bpp = 0;
		int blendMode = 0;
	};
}
