"""Contact-aware admission for a verified handoff within one command writer."""

from .direct_dds import vector


def handoff_target_class(base, reference):
    """Use the retained arm command only for a trusted, continuous loaded handoff."""
    reference = vector(reference, "handoff command reference").copy()
    reference.setflags(write=False)

    class HandoffTarget(base):
        def check_start(self, goal, measured):
            goal = vector(goal, "handoff goal")
            measured = vector(measured, "measured")
            start_reference = measured.copy()
            arm_axes = [axis for axis in self.axes if 15 <= axis < 29]
            start_reference[arm_axes] = reference[arm_axes]
            super().check_start(goal, start_reference)
            self.check_tracking(measured, goal)

    return HandoffTarget


def configure_loaded_arrival(reset):
    """Require measured waist arrival while retaining all-axis tracking guards."""
    reset.arrival_axes = tuple(axis for axis in reset.target.axes if 0 <= axis < 3)
    reset.loaded_handoff = True
    reset.events.append({
        "event": "loaded_handoff_arrival",
        "completion_criterion": "waist_measured_arrival_arm_command_complete",
        "arrival_axes": list(reset.arrival_axes),
        "grasp_verified": False,
    })
    return reset
