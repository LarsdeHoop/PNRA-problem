import json
from HelperFunctions import shift_day_to_index, instance_name_to_dataset


class Instance:
    def __init__(self, file_path, print_instance_info = False):
        with open(file_path, 'r') as file:
            instance_dict = json.load(file)

        self.weights = {"Continuity": 1,
                        "Gender-Mixing": 1,
                        "Skill Requirements": 1,
                        "Workload Violation": 1,
                        "Workload Imbalance": 1}

        # General data
        self.instance_name = instance_dict["instance_name"]
        self.numberOfDays = instance_dict["number_of_days"]
        self.skillLevels = instance_dict["skill_levels"]
        self.shiftTypes = instance_dict["shift_types"]
        self.ageGroups = instance_dict["age_groups"]

        # Some backup error checks
        if len(self.shiftTypes) != 3 or self.shiftTypes != ["early", "late", "night"]:
            raise Exception(f"A different number of shift types. shiftTypes = {self.shiftTypes}")

        # Shifts
        self.all_shifts = list(range(3 * self.numberOfDays))
        self.early_shifts = [3 * i for i in range(self.numberOfDays)]
        self.late_shifts = [3 * i + 1 for i in range(self.numberOfDays)]
        self.night_shifts = [3 * i + 2 for i in range(self.numberOfDays)]

        # Patients (incl. occupants)
        self.patient_ids = []
        self.occupant_ids = []
        self.non_occupant_ids = []
        self.patients = dict()
        for patient in instance_dict["patients"]:
            patient_id = patient["id"]
            gender = patient["gender"]
            age_group = patient["age_group"]
            admission_day = patient["admission_day"]
            length_of_stay = patient["length_of_stay"]
            workload_produced = patient["workload_produced"]
            skill_level_required = patient["skill_level_required"]
            prev_room = patient["room_id"]
            incompatible_rooms = patient["incompatible_room_ids"]

            shifts = []
            workload = dict()
            skill = dict()
            starting_shift = shift_day_to_index(admission_day, self.shiftTypes,"early")
            for i in range(3*length_of_stay):
                shift_index = starting_shift + i
                if shift_index >= 3 * self.numberOfDays:  # when patients are discharged after time horizon
                    break
                shifts.append(shift_index)
                workload[shift_index] = workload_produced[i]
                skill[shift_index] = skill_level_required[i]

            self.patient_ids.append(patient_id)
            if prev_room is not None:
                self.occupant_ids.append(patient_id)
            else:
                self.non_occupant_ids.append(patient_id)
            self.patients[patient_id] = {"gender": gender,
                                         "age_group": age_group,
                                         "prev_room": prev_room,
                                         "incompatible_rooms": incompatible_rooms,
                                         "shifts": shifts,
                                         "workload":workload,
                                         "skill":skill}
        self.patients_per_shift = {s: [] for s in self.all_shifts}
        for p in self.patient_ids:
            for s in self.patients[p]["shifts"]:
                self.patients_per_shift[s].append(p)

        # Nurses
        self.nurse_ids = []
        self.nurses = dict()
        for nurse in instance_dict["nurses"]:
            nurse_id = nurse["id"]
            skill_level = nurse["skill_level"]
            working_shifts = nurse["working_shifts"]

            shifts = []
            max_load = dict()
            for shift in working_shifts:
                shift_index = shift_day_to_index(shift["day"], self.shiftTypes, shift["shift"])
                shifts.append(shift_index)
                max_load[shift_index] = shift["max_load"]
            self.nurse_ids.append(nurse_id)
            self.nurses[nurse_id] = {"skill": skill_level, "shifts": shifts,
                                     "max_load": max_load}

        self.nurses_per_shift = {s: [] for s in self.all_shifts}
        for n in self.nurse_ids:
            for s in self.nurses[n]["shifts"]:
                self.nurses_per_shift[s].append(n)

        # Rooms
        self.room_ids = []
        self.room_capacities = dict()
        for room in instance_dict["rooms"]:
            room_id = room["id"]
            capacity = room["capacity"]
            self.room_ids.append(room_id)
            self.room_capacities[room_id] = capacity

        if print_instance_info:
            print(f"\nRetrieving instance data successful. Instance {self.instance_name} contains the following")
            print(f"{self.numberOfDays} days")
            print(f"{len(self.room_ids)} rooms")
            male_occ = len([o for o in self.occupant_ids if self.patients[o]['gender'] == 'A'])
            print(f"{len(self.occupant_ids)} occupants ({male_occ} male and {len(self.occupant_ids) - male_occ} female)")
            male_pat = len([p for p in self.patient_ids if self.patients[p]['gender'] == 'A']) - male_occ
            print(f"{len(self.patient_ids) - len(self.occupant_ids)} new patients "
                  f"({male_pat} male and {len(self.patient_ids) - len(self.occupant_ids) - male_pat} female)")
            print(f"{len(self.nurse_ids)} nurses")

if __name__ == "__main__":
    instance_name = "test01"
    dataset_name = instance_name_to_dataset(instance_name)
    file_path = "Instances/" + dataset_name + "/" + instance_name + ".json"
    instance = Instance(file_path, print_instance_info=True)

