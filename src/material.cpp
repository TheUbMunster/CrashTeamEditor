#include "material.h"

template class MaterialProperty<std::string>;
template class MaterialProperty<uint16_t>;
template class MaterialProperty<bool>;

template<typename T>
MaterialProperty<T>::MaterialProperty()
{
	m_preview = std::unordered_map<std::string, T>();
	m_backup = std::unordered_map<std::string, T>();
}

template<typename T>
void MaterialProperty<T>::SetPreview(const std::string& material, const T& preview)
{
	m_preview[material] = preview;
	m_materialsChanged.push_back(material);
}

template<typename T>
void MaterialProperty<T>::SetBackup(const std::string& material, const T& backup)
{
	m_backup[material] = backup;
}

template<typename T>
void MaterialProperty<T>::SetDefaultValue(const std::string& material, const T& value)
{
	m_preview[material] = value;
	m_backup[material] = value;
}

template<typename T>
void MaterialProperty<T>::Restore()
{
	if (m_materialsChanged.empty()) { return; }

	for (const std::string& material : m_materialsChanged)
	{
		m_preview[material] = m_backup[material];
	}
	m_materialsChanged.clear();
}

template<typename T>
void MaterialProperty<T>::Clear()
{
	m_preview.clear();
	m_backup.clear();
	m_materialsChanged.clear();
}

template<typename T>
T& MaterialProperty<T>::GetPreview(const std::string& material)
{
	return m_preview[material];
}
