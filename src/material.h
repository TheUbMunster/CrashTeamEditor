#pragma once

#include <unordered_map>
#include <string>

class MaterialBase
{
public:
	virtual void Restore() = 0;
	virtual void Clear() = 0;
};

template <typename T>
class MaterialProperty : public MaterialBase
{
public:
	MaterialProperty();
	void SetPreview(const std::string& material, const T& preview);
	void SetBackup(const std::string& material, const T& backup);
	void SetDefaultValue(const std::string& material, const T& value);
	void Restore() override;
	void Clear() override;
	T& GetPreview(const std::string& material);

private:
	std::unordered_map<std::string, T> m_backup;
	std::unordered_map<std::string, T> m_preview;
	std::vector<std::string> m_materialsChanged;
};

void ClearMaterials();
void RestoreMaterials();