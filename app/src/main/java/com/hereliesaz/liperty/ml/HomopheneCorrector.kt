package com.HereLiesAz.liperty.ml

import android.content.Context
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader

class HomopheneCorrector(private val context: Context) {

    private val homopheneMap: Map<String, List<String>> by lazy {
        loadHomopheneMap()
    }

    private fun loadHomopheneMap(): Map<String, List<String>> {
        val map = mutableMapOf<String, List<String>>()
        try {
            val inputStream = try {
                context.assets.open("homophones.json")
            } catch (e: Exception) {
                // Fallback for tests: Load from classpath
                javaClass.classLoader?.getResourceAsStream("homophones.json")
                    ?: throw Exception("homophones.json not found in assets or resources")
            }

            val reader = BufferedReader(InputStreamReader(inputStream))
            val jsonString = reader.use { it.readText() }
            val jsonObject = JSONObject(jsonString)

            val iterator = jsonObject.keys()
            while (iterator.hasNext()) {
                val key = iterator.next()
                val jsonArray = jsonObject.getJSONArray(key)
                val alternatives = mutableListOf<String>()
                for (i in 0 until jsonArray.length()) {
                    alternatives.add(jsonArray.getString(i))
                }
                map[key] = alternatives
            }
        } catch (e: Exception) {
            e.printStackTrace()
            // Fallback or empty map if loading fails
        }
        return map
    }

    /**
     * Returns a list of alternative words that are homophenes of the input word.
     */
    fun getAlternatives(word: String): List<String> {
        val lowerCaseWord = word.lowercase()
        return homopheneMap[lowerCaseWord] ?: emptyList()
    }

    /**
     * More advanced: Check if two words are homophenes based on phoneme mapping.
     */
    fun areHomophenes(word1: String, word2: String): Boolean {
        val alts = getAlternatives(word1)
        return alts.contains(word2.lowercase())
    }
}
