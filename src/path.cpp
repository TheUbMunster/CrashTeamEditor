#include "path.h"

#include <algorithm>

Path::Path(size_t index)
{
	m_index = index;
	m_start = 0;
	m_end = 0;
	m_left = nullptr;
	m_right = nullptr;
	m_previewValueStart = 0;
	m_previewLabelStart = std::string();
	m_quadIndexesStart = std::vector<size_t>();
	m_previewValueEnd = 0;
	m_previewLabelEnd = std::string();
	m_quadIndexesEnd = std::vector<size_t>();
	m_previewValueIgnore = 0;
	m_previewLabelIgnore = std::string();
	m_quadIndexesIgnore = std::vector<size_t>();
}

Path::~Path()
{
	delete m_left;
	delete m_right;
	m_left = nullptr;
	m_right = nullptr;
}

size_t Path::Index() const
{
	return m_index;
}

size_t Path::Start() const
{
	return m_start;
}

size_t Path::End() const
{
	return m_end;
}

bool Path::Ready() const
{
	bool left = true; if (m_left) { left = m_left->Ready(); }
	bool right = true; if (m_right) { right = m_right->Ready(); }
	return left && right && !m_quadIndexesStart.empty() && !m_quadIndexesEnd.empty();
}

void Path::UpdateDist(float dist, const Vec3& refPoint, std::vector<Checkpoint>& checkpoints)
{
	dist += (refPoint - checkpoints[m_end].Pos()).Length();
	size_t currIndex = m_end;
	while (true)
	{
		Checkpoint& currCheckpoint = checkpoints[currIndex];
		currCheckpoint.UpdateDistFinish(dist + currCheckpoint.DistFinish());
		if (currIndex == m_start) { break; }
		currIndex = currCheckpoint.Down();
	}
	if (m_left) { m_left->UpdateDist(dist, checkpoints[m_end].Pos(), checkpoints); }
	if (m_right) { m_right->UpdateDist(dist, checkpoints[m_end].Pos(), checkpoints); }
}

std::vector<Checkpoint> Path::GeneratePath(size_t pathStartIndex, std::vector<Quadblock>& quadblocks)
{
	/*
		Begin from the start point, find all neighbour quadblocks.
		Find the midpoint of the quad group, then find the closest vertex to
		this midpoint. Repeat this process until you're neighboring the end path.
	*/

	size_t visitedCount = 0;
	std::vector<size_t> startEndIndexes;
	std::vector<bool> visitedQuadblocks(quadblocks.size(), false);
	GetStartEndIndexes(startEndIndexes);
	for (const size_t index : startEndIndexes)
	{
		visitedQuadblocks[index] = true;
		visitedCount++;
	}

	for (size_t i = 0; i < quadblocks.size(); i++)
	{
		if (!quadblocks[i].CheckpointStatus())
		{
			visitedQuadblocks[i] = true;
			visitedCount++;
		}
	}

	std::vector<size_t> currQuadblocks = m_quadIndexesStart;
	std::vector<std::vector<size_t>> quadIndexesPerChunk;
	while (true)
	{
		std::vector<size_t> nextQuadblocks;
		if (visitedCount < quadblocks.size())
		{
			for (const size_t index : currQuadblocks)
			{
				for (size_t i = 0; i < quadblocks.size(); i++)
				{
					if (!visitedQuadblocks[i] && quadblocks[index].Neighbours(quadblocks[i]))
					{
						nextQuadblocks.push_back(i);
						visitedQuadblocks[i] = true;
						visitedCount++;
					}
				}
			}
		}
		quadIndexesPerChunk.push_back(currQuadblocks);
		if (nextQuadblocks.empty()) { break; }
		currQuadblocks = nextQuadblocks;
	}
	quadIndexesPerChunk.push_back(m_quadIndexesEnd);

	Vec3 lastChunkVertex;
	float distFinish = 0.0f;
	std::vector<float> distFinishes;
	std::vector<Checkpoint> checkpoints;
	int currCheckpointIndex = static_cast<int>(pathStartIndex);
	for (const std::vector<size_t>& quadIndexSet : quadIndexesPerChunk)
	{
		BoundingBox bbox;
		bbox.min = Vec3(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
		bbox.max = Vec3(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
		for (const size_t index : quadIndexSet)
		{
			const BoundingBox& quadBbox = quadblocks[index].GetBoundingBox();
			bbox.min.x = std::min(bbox.min.x, quadBbox.min.x); bbox.max.x = std::max(bbox.max.x, quadBbox.max.x);
			bbox.min.y = std::min(bbox.min.y, quadBbox.min.y); bbox.max.y = std::max(bbox.max.y, quadBbox.max.y);
			bbox.min.z = std::min(bbox.min.z, quadBbox.min.z); bbox.max.z = std::max(bbox.max.z, quadBbox.max.z);
		}

		Vec3 chunkVertex;
		size_t chunkQuadIndex = 0;
		Vec3 chunkCenter = bbox.Midpoint();
		float closestDist = std::numeric_limits<float>::max();
		for (const size_t index : quadIndexSet)
		{
			Vec3 closestVertex;
			float dist = quadblocks[index].DistanceClosestVertex(closestVertex, chunkCenter);
			if (dist < closestDist)
			{
				closestDist = dist;
				chunkVertex = closestVertex;
				chunkQuadIndex = index;
			}
			quadblocks[index].SetCheckpoint(currCheckpointIndex);
		}
		if (!checkpoints.empty()) { distFinish += (lastChunkVertex - chunkVertex).Length(); }
		distFinishes.push_back(distFinish);
		checkpoints.emplace_back(currCheckpointIndex, chunkVertex, quadblocks[chunkQuadIndex].Name());
		checkpoints.back().UpdateUp(currCheckpointIndex + 1);
		checkpoints.back().UpdateDown(currCheckpointIndex - 1);
		currCheckpointIndex++;
		lastChunkVertex = chunkVertex;
	}

	m_start = pathStartIndex;
	m_end = m_start + checkpoints.size() - 1;

	size_t j = 0;
	for (int i = static_cast<int>(checkpoints.size()) - 1; i >= 0; i--)
	{
		checkpoints[j++].UpdateDistFinish(distFinishes[i]);
	}

	checkpoints.front().UpdateDown(NONE_CHECKPOINT_INDEX);
	checkpoints.back().UpdateUp(NONE_CHECKPOINT_INDEX);

	pathStartIndex += checkpoints.size();
	std::vector<Checkpoint> leftCheckpoints, rightCheckpoints;
	if (m_left)
	{
		leftCheckpoints = m_left->GeneratePath(pathStartIndex, quadblocks);
		checkpoints.back().UpdateLeft(leftCheckpoints.back().Index());
		checkpoints.front().UpdateLeft(leftCheckpoints.front().Index());
		leftCheckpoints.back().UpdateRight(checkpoints.back().Index());
		leftCheckpoints.front().UpdateRight(checkpoints.front().Index());
		pathStartIndex += leftCheckpoints.size();
	}
	if (m_right)
	{
		rightCheckpoints = m_right->GeneratePath(pathStartIndex, quadblocks);
		checkpoints.back().UpdateRight(rightCheckpoints.back().Index());
		checkpoints.front().UpdateRight(rightCheckpoints.front().Index());
		rightCheckpoints.back().UpdateLeft(checkpoints.back().Index());
		rightCheckpoints.front().UpdateLeft(checkpoints.front().Index());
	}

	for (const Checkpoint& checkpoint : leftCheckpoints) { checkpoints.push_back(checkpoint); }
	for (const Checkpoint& checkpoint : rightCheckpoints) { checkpoints.push_back(checkpoint); }

	return checkpoints;
}

void Path::GetStartEndIndexes(std::vector<size_t>& out) const
{
	if (m_left) { m_left->GetStartEndIndexes(out); }
	if (m_right) { m_right->GetStartEndIndexes(out); }
	for (const size_t index : m_quadIndexesStart) { out.push_back(index); }
	for (const size_t index : m_quadIndexesIgnore) { out.push_back(index); }
	for (const size_t index : m_quadIndexesEnd) { out.push_back(index); }
}