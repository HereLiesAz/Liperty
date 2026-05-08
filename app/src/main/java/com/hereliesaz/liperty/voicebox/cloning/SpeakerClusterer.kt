package com.hereliesaz.liperty.voicebox.cloning

import kotlin.math.sqrt

/**
 * Clusters speech segments by speaker identity using agglomerative clustering
 * on speaker embeddings (cosine similarity).
 *
 * Designed for the voice cloning import flow: when a home video has multiple
 * speakers, this identifies distinct speakers so the user can pick "which is me."
 */
class SpeakerClusterer {

    data class SpeakerCluster(
        val clusterId: Int,
        val segmentIndices: List<Int>,
        val segments: List<AudioPreprocessor.SpeechSegment>,
        val embeddings: List<FloatArray>,
        val centroidEmbedding: FloatArray,
        val totalSpeechDurationMs: Long,
        /** Longest, highest-SNR segment — used for "play sample" in the UI. */
        val representativeSegment: AudioPreprocessor.SpeechSegment
    )

    data class ClusteringResult(
        val clusters: List<SpeakerCluster>,
        /** True if all segments belong to a single speaker. */
        val singleSpeaker: Boolean
    )

    /**
     * Clusters speech segments by speaker identity.
     *
     * @param segments Speech segments from [AudioPreprocessor].
     * @param embeddings Per-segment speaker embeddings (same order).
     * @param similarityThreshold Cosine similarity above which two segments are
     *        considered the same speaker. Default 0.75 (conservative).
     */
    fun cluster(
        segments: List<AudioPreprocessor.SpeechSegment>,
        embeddings: List<FloatArray>,
        similarityThreshold: Float = 0.75f
    ): ClusteringResult {
        require(segments.size == embeddings.size) {
            "Segments (${segments.size}) and embeddings (${embeddings.size}) must be same length"
        }

        if (segments.isEmpty()) {
            return ClusteringResult(emptyList(), singleSpeaker = true)
        }
        if (segments.size == 1) {
            val cluster = buildCluster(0, listOf(0), segments, embeddings)
            return ClusteringResult(listOf(cluster), singleSpeaker = true)
        }

        val clusterAssignments = agglomerativeClustering(embeddings, similarityThreshold)

        val clusters = clusterAssignments.mapIndexed { idx, indices ->
            buildCluster(idx, indices, segments, embeddings)
        }.sortedByDescending { it.totalSpeechDurationMs }

        return ClusteringResult(
            clusters = clusters,
            singleSpeaker = clusters.size <= 1
        )
    }

    /**
     * Cosine similarity between two embedding vectors.
     * Returns value in [-1, 1]; 1 = identical direction.
     */
    fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        require(a.size == b.size) { "Vectors must be same dimension" }
        var dot = 0.0
        var normA = 0.0
        var normB = 0.0
        for (i in a.indices) {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        val denom = sqrt(normA) * sqrt(normB)
        return if (denom < 1e-10) 0f else (dot / denom).toFloat()
    }

    /**
     * Bottom-up agglomerative clustering with average-linkage.
     * Each segment starts as its own cluster. The two closest clusters are
     * merged repeatedly until no pair exceeds [threshold].
     *
     * @return List of clusters, each a list of original segment indices.
     */
    private fun agglomerativeClustering(
        embeddings: List<FloatArray>,
        threshold: Float
    ): List<List<Int>> {
        val n = embeddings.size

        // Initialize: each segment is its own cluster
        val clusters = (0 until n).map { mutableListOf(it) }.toMutableList()
        // Centroids for average-linkage
        val centroids = embeddings.map { it.copyOf() }.toMutableList()

        while (clusters.size > 1) {
            // Find most similar pair
            var bestSim = -1f
            var bestI = -1
            var bestJ = -1
            for (i in clusters.indices) {
                for (j in i + 1 until clusters.size) {
                    val sim = cosineSimilarity(centroids[i], centroids[j])
                    if (sim > bestSim) {
                        bestSim = sim
                        bestI = i
                        bestJ = j
                    }
                }
            }

            if (bestSim < threshold) break // No pair is similar enough

            // Merge bestJ into bestI
            val mergedIndices = clusters[bestI] + clusters[bestJ]
            val mergedEmbeddings = mergedIndices.map { embeddings[it] }
            val mergedCentroid = computeCentroid(mergedEmbeddings)

            clusters[bestI] = mergedIndices.toMutableList()
            centroids[bestI] = mergedCentroid

            clusters.removeAt(bestJ)
            centroids.removeAt(bestJ)
        }

        return clusters
    }

    /**
     * Element-wise mean of a set of embeddings.
     */
    fun computeCentroid(embeddings: List<FloatArray>): FloatArray {
        if (embeddings.isEmpty()) return floatArrayOf()
        val dim = embeddings[0].size
        val centroid = FloatArray(dim)
        for (emb in embeddings) {
            for (i in centroid.indices) centroid[i] += emb[i]
        }
        for (i in centroid.indices) centroid[i] /= embeddings.size
        return centroid
    }

    private fun buildCluster(
        id: Int,
        indices: List<Int>,
        allSegments: List<AudioPreprocessor.SpeechSegment>,
        allEmbeddings: List<FloatArray>
    ): SpeakerCluster {
        val clusterSegments = indices.map { allSegments[it] }
        val clusterEmbeddings = indices.map { allEmbeddings[it] }
        val centroid = computeCentroid(clusterEmbeddings)

        // Representative: prefer longest segment with best SNR
        val representative = clusterSegments.maxByOrNull {
            it.durationMs * 0.6 + it.snrEstimate * 0.4
        } ?: clusterSegments.first()

        return SpeakerCluster(
            clusterId = id,
            segmentIndices = indices,
            segments = clusterSegments,
            embeddings = clusterEmbeddings,
            centroidEmbedding = centroid,
            totalSpeechDurationMs = clusterSegments.sumOf { it.durationMs },
            representativeSegment = representative
        )
    }
}
