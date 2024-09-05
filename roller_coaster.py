class RollerCoaster:
    def __init__(self, coaster_type, max_speed, bumps_per_second, lift_type):
        self.coaster_type = coaster_type
        self.max_speed = max_speed
        self.bumps_per_second = bumps_per_second
        self.lift_type = lift_type
        self.scale_factor = self.get_scale_factor()

    def get_scale_factor(self):
        # Define scale factors for each coaster type
        if self.coaster_type == "Wooden":
            return 1.0
        elif self.coaster_type == "Steel":
            return 2.0
        elif self.coaster_type == "Suspended":
            return 0.5
        else:
            raise ValueError("Unknown Roller Coaster Type")

    def comfort(self):
        if self.coaster_type == "Steel":
            return min(1.0, 1.0 / self.bumps_per_second, 5 / self.max_speed)
        elif self.coaster_type == "Suspended":
            comfort = min(1.0, 1.0 / self.bumps_per_second, 1 / self.max_speed)
            if self.lift_type == "Cable":
                comfort += 0.5
            return comfort
        else:  # Wooden coaster
            return min(1.0, 1.0 / self.bumps_per_second, 1.0 / self.max_speed)

    def overall_score(self):
        return self.scale_factor * self.comfort() * self.max_speed


def calculate_scores(coasters):
    for coaster_data in coasters:
        coaster_type, max_speed, bumps_per_second, lift_type = coaster_data.split()
        max_speed = float(max_speed)
        bumps_per_second = float(bumps_per_second)

        coaster = RollerCoaster(coaster_type, max_speed, bumps_per_second, lift_type)
        overall_score = coaster.overall_score()
        print(f"Overall: {overall_score:.1f}")


# Example input
coasters = [
    "Wooden 4 1.0 Chain",
    "Steel 20 2.0 Cable",
    "Suspended 2 1.5 Cable",
    "Suspended 2 1.5 Chain"
]

# Calculate and print overall scores
calculate_scores(coasters)
