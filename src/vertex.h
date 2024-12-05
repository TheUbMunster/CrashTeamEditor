#pragma once

#include "math.h"

#include <cstdint>

enum class VertexFlags : uint16_t
{
	NONE = 0,
};

class Vertex
{
public:
	Vertex();
	Vertex(const Vec3& pos);
	bool IsEdited() const;
	void RenderUI(size_t index);
	std::vector<uint8_t> Serialize() const;
	inline bool operator==(const Vertex& v) const { return (m_pos == v.m_pos) && (m_flags == v.m_flags) && (m_colorHigh == v.m_colorHigh) && (m_colorLow == v.m_colorLow); };

public:
	Vec3 m_pos;

private:
	VertexFlags m_flags;
	Color m_colorHigh;
	Color m_colorLow;
	bool m_editedPos;

	friend std::hash<Vertex>;
};

template<>
struct std::hash<Vertex>
{
	inline std::size_t operator()(const Vertex& key) const
	{
		return ((((std::hash<Vec3>()(key.m_pos) ^ (std::hash<VertexFlags>()(key.m_flags) << 1)) >> 1) ^ (std::hash<Color>()(key.m_colorHigh) << 1)) >> 2) ^ (std::hash<Color>()(key.m_colorLow) << 2);
	}
};