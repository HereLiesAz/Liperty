package com.hereliesaz.liperty.ml

import android.content.Context
import java.io.File

/**
 * Parses Auto-AVSR / ESPnet-style token list files (e.g. `unigram5000_units.txt`)
 * for use with [SubwordCTCDecoder].
 *
 * File format is forgiving — every line of the form `<token>[whitespace<index>]`
 * is accepted. The index column is ignored (the token's position in the
 * returned list IS the index). Lines that are blank after trimming are skipped.
 *
 * The blank token (CTC index 0) is **not** in the file. Use [withBlankAtZero]
 * to prepend it before passing the list to a CTC decoder.
 */
object VocabularyLoader {

    /**
     * Reads `assetName` from [context]'s assets and returns the parsed token
     * list with a blank prepended at index 0 — the form a [SubwordCTCDecoder]
     * directly consumes.
     */
    fun loadFromAssets(context: Context, assetName: String, blank: String = "_"): List<String> {
        val text = context.assets.open(assetName).bufferedReader().use { it.readText() }
        return withBlankAtZero(parseTokenList(text), blank)
    }

    /**
     * Reads `name` preferring [Context.getFilesDir] (where [ModelDownloadManager]
     * places files at first launch) and falling back to bundled assets (dev
     * builds). Returns the parsed token list with a blank prepended at index 0.
     *
     * Use this instead of [loadFromAssets] for vocabularies that are downloaded
     * rather than bundled — the assets-only path can't see a downloaded file.
     */
    fun load(context: Context, name: String, blank: String = "_"): List<String> =
        withBlankAtZero(parseTokenList(readTextPreferringFiles(context, name)), blank)

    /**
     * Raw text read that prefers a downloaded file in [Context.getFilesDir] over
     * a bundled asset. Throws if the resource is in neither location.
     */
    fun readTextPreferringFiles(context: Context, name: String): String {
        val file = File(context.filesDir, name)
        if (file.exists() && file.length() > 0) return file.readText()
        return context.assets.open(name).bufferedReader().use { it.readText() }
    }

    /**
     * Pure parse step. Strip the optional trailing index column, drop blank
     * lines.
     */
    fun parseTokenList(text: String): List<String> {
        val out = ArrayList<String>()
        for (raw in text.lineSequence()) {
            val trimmed = raw.trim()
            if (trimmed.isEmpty()) continue
            // Split on whitespace; first column is the token, rest is the
            // optional index. Tokens themselves never contain whitespace
            // (SentencePiece marks word boundaries with `▁`, not spaces).
            val firstWs = trimmed.indexOfFirst { it == ' ' || it == '\t' }
            val token = if (firstWs < 0) trimmed else trimmed.substring(0, firstWs)
            if (token.isEmpty()) continue
            out.add(token)
        }
        return out
    }

    /**
     * Prepend [blank] at index 0 unless the list already starts with it. The
     * resulting list maps directly to a CTC head whose blank index is 0.
     */
    fun withBlankAtZero(tokens: List<String>, blank: String = "_"): List<String> {
        if (tokens.isNotEmpty() && tokens[0] == blank) return tokens
        val out = ArrayList<String>(tokens.size + 1)
        out.add(blank)
        out.addAll(tokens)
        return out
    }
}
