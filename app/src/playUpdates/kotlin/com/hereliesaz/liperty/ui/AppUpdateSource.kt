package com.hereliesaz.liperty.ui

import android.app.Activity
import com.hereliesaz.aznavrail.updates.AzUpdateChecker
import com.hereliesaz.aznavrail.updates.play.AzPlayUpdates

/** Explicit Settings action, separate from recognition and consent. */
internal fun checkAppUpdate(activity: Activity) {
    AzUpdateChecker.check(activity, "HereLiesAz/Liperty", manual = true, playSource = AzPlayUpdates)
}
