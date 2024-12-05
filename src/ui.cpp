#include "ui.h"

#include <imgui.h>
#include <portable-file-dialogs.h>

#include <filesystem>

void UI::Render(int width, int height)
{
	MainMenu();
	m_lev.RenderUI();
}

void UI::MainMenu()
{
	if (ImGui::BeginMainMenuBar())
	{
		if (ImGui::BeginMenu("File"))
		{
			if (ImGui::MenuItem("Open"))
			{
				auto selection = pfd::open_file("Level File").result();
				if (!selection.empty())
				{
					const std::filesystem::path levPath = selection.front();
					m_lev.Load(levPath);
				}
			}
			if (ImGui::MenuItem("Save"))
			{
				auto selection = pfd::select_folder("Level Folder").result();
				if (!selection.empty())
				{
					const std::filesystem::path path = selection + "\\";
					m_lev.Save(path);
				}
			}
			ImGui::EndMenu();
		}
		ImGui::EndMainMenuBar();
	}
}