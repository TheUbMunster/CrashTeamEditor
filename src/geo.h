#pragma once

#include "utils.h"

#include <vector>
#include <cstdint>
#include <cmath>

struct Color
{
	Color() : rb(0u), gb(0u), bb(0u), a(false) {};
	Color(float r, float g, float b) : rb(static_cast<unsigned char>(Clamp(r * 255.0f, 0.0f, 255.0f))), gb(static_cast<unsigned char>(Clamp(g * 255.0f, 0.0f, 255.0f))), bb(static_cast<unsigned char>(Clamp(b * 255.0f, 0.0f, 255.0f))), a(false) {};
	Color(unsigned char r, unsigned char g, unsigned char b) : a(false)
	{
		this->rb = r;
		this->gb = g;
		this->bb = b;
	};
	Color(double hue, double sat, double value) : a(false) {
		double      hh, p, q, t, ff;
		long        i;

		if (sat == 0) {
			this->rb = this->gb = this->bb = 0;
			return;
		}
		else {
			hh = hue;
			if (hh >= 360.0) hh = 0.0;
			hh /= 60.0;
			i = (long)hh;
			ff = hh - i;
			p = value * (1.0 - sat);
			q = value * (1.0 - (sat * ff));
			t = value * (1.0 - (sat * (1.0 - ff)));

			t = Clamp(t, 0.0, 1.0);
			p = Clamp(p, 0.0, 1.0);
			q = Clamp(q, 0.0, 1.0);
			value = Clamp(value, 0.0, 1.0);

			switch (i) {
			case 0:
				this->rb = static_cast<unsigned char>(value * 255.0f);
				this->gb = static_cast<unsigned char>(t * 255.0f);
				this->bb = static_cast<unsigned char>(p * 255.0f);
				break;
			case 1:
				this->rb = static_cast<unsigned char>(q * 255.0f);
				this->gb = static_cast<unsigned char>(value * 255.0f);
				this->bb = static_cast<unsigned char>(p * 255.0f);
				break;
			case 2:
				this->rb = static_cast<unsigned char>(p * 255.0f);
				this->gb = static_cast<unsigned char>(value * 255.0f);
				this->bb = static_cast<unsigned char>(t * 255.0f);
				break;
			case 3:
				this->rb = static_cast<unsigned char>(p * 255.0f);
				this->gb = static_cast<unsigned char>(q * 255.0f);
				this->bb = static_cast<unsigned char>(value * 255.0f);
				break;
			case 4:
				this->rb = static_cast<unsigned char>(t * 255.0f);
				this->gb = static_cast<unsigned char>(p * 255.0f);
				this->bb = static_cast<unsigned char>(value * 255.0f);
				break;
			case 5:
			default:
				this->rb = static_cast<unsigned char>(value * 255.0f);
				this->gb = static_cast<unsigned char>(p * 255.0f);
				this->bb = static_cast<unsigned char>(q * 255.0f);
				break;
			}
		}
	}
	//float* Data() { return &r; }
	inline bool operator==(const Color& color) const { return (rb == color.rb) && (gb == color.gb) && (bb == color.bb) && (a == color.a); }

	unsigned char rb, gb, bb;
	bool a;

	float RAsFloat() const { return static_cast<float>(rb) / 255.0f; }
	float GAsFloat() const { return static_cast<float>(gb) / 255.0f; }
	float BAsFloat() const { return static_cast<float>(bb) / 255.0f; }
};

template<>
struct std::hash<Color>
{
	inline std::size_t operator()(const Color& key) const
	{
		return ((((std::hash<float>()(key.RAsFloat()) ^ (std::hash<float>()(key.GAsFloat()) << 1)) >> 1) ^ (std::hash<float>()(key.BAsFloat()) << 1)) >> 2) ^ (std::hash<bool>()(key.a) << 2);
	}
};

struct Vec3
{
	Vec3() : x(0.0f), y(0.0f), z(0.0f) {};
	Vec3(float x, float y, float z) : x(x), y(y), z(z) {};
	float* Data() { return &x; }
	inline float Length() const { return static_cast<float>(std::sqrt((x * x) + (y * y) + (z * z))); }
	inline Vec3 Cross(const Vec3& v) const { return { y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - v.x * y }; }

	inline Vec3 operator+(const Vec3& v) const { return { x + v.x, y + v.y, z + v.z }; }
	inline Vec3 operator-(const Vec3& v) const { return { x - v.x, y - v.y, z - v.z }; }
	inline Vec3 operator*(float n) const { return { x * n, y * n, z * n }; }
	inline Vec3 operator/(float n) const { return { x / n, y / n, z / n }; }
	inline bool operator>(float n) const { return x > n && y > n && z > n; }
	inline bool operator<(float n) const { return x < n && y < n && z < n; }
	inline bool operator==(const Vec3& v) const { return (x == v.x) && (y == v.y) && (z == v.z); }
	inline Vec3& operator+=(const Vec3& v) { x += v.x; y += v.y; z += v.z; return *this; }
	inline Vec3& operator-=(const Vec3& v) { x -= v.x; y -= v.y; z -= v.z; return *this; }
	inline Vec3& operator*=(float n) { x *= n; y *= n; z *= n; return *this; }
	inline Vec3& operator/=(float n) { x /= n; y /= n; z /= n; return *this; }

	float x;
	float y;
	float z;
};

template<>
struct std::hash<Vec3>
{
	inline std::size_t operator()(const Vec3& key) const
	{
		return ((std::hash<float>()(key.x) ^ (std::hash<float>()(key.y) << 1)) >> 1) ^ (std::hash<float>()(key.z) << 1);
	}
};

struct BoundingBox
{
	Vec3 min;
	Vec3 max;

	float Area() const;
	float SemiPerimeter() const;
	Vec3 AxisLength() const;
	Vec3 Midpoint() const;
	void RenderUI() const;
};

struct Point
{
	Vec3 pos;
	Vec3 normal;
	Color color;

	Point() {};
	Point(float x, float y, float z)
	{
		pos = Vec3(x, y, z);
		color = Color((unsigned char)255u, 255u, 255u);
		normal = Vec3();
	};
	Point(float x, float y, float z, unsigned char r, unsigned char g, unsigned char b)
	{
		pos = Vec3(x, y, z);
		color = Color(r, g, b);
		normal = Vec3();
	};
};

struct Tri
{
	Tri() : p() {};
	Tri(const Point& p0, const Point& p1, const Point& p2);

	Point p[3];
};

struct Quad
{
	Quad() : p() {};
	Quad(const Point& p0, const Point& p1, const Point& p2, const Point& p3);

	Point p[4];
};