import numpy as np
from HelperFunctions import *
from ComputeObjective import compute_objective_LS, compute_objective
from SimAnnealingFunctions import *
import matplotlib.pyplot as plt
import datetime

class SimulatedAnnealer:
    def __init__(self, instance):
        self.instance = instance

        ### PARAMETERS
        # Iterations
        self.cur_iter = 0
        self.max_iters = 100000

        # temperature parameters
        self.T_start = 2
        self.T_end = 0.06
        self.cur_temperature = self.T_start
        self.cooling_rate = (self.T_end / self.T_start) ** (1 / self.max_iters)

        # move parameters
        self.move_options = ["MovePatient","SwapPatients","ChangeNurse","RemoveNurse"]
        self.move_prob_vector =  (0.2,0.1,0.6,0.1) # must sum to 1
        self.move_prob_vector_if_fail = (2/9,1/9,6/9,0)

        # other parameters
        self.do_objective_check = False
        self.store_best_solution = False # set to False to prevent storing the best solution.
                                        # Improves speed (especially when using random initialization)

        # initial solution
        self.initial_solution_file_name = None
        self.initial_solution_path_name = None

        ### SOLUTION AND OBJECTIVE
        # Solutions
        self.initial_solution = None
        self.cur_solution = None
        self.best_solution = None

        # Solution attributes
        self.solution_attributes = None

        # Objectives
        self.cur_obj = float('inf')
        self.best_obj = float('inf')

        ### RECORDING PROCESSES
        # Record the process of the objective
        self.cur_obj_process = []
        self.best_obj_process = []
        self.move_count = {move_type:0 for move_type in self.move_options}

        ### LOOKUP THINGS FOR SPEED
        feasible_room_shift_pairs = []
        for r in instance.room_ids:
            for s in instance.all_shifts:
                if len(instance.nurses_per_shift[s]) >= 2:
                    feasible_room_shift_pairs.append((r, s))
        self.feasible_room_shift_pairs = feasible_room_shift_pairs

        other_nurses_per_shift = {}
        for s0 in instance.all_shifts:
            for n1_ in instance.nurses_per_shift[s0]:
                other_nurses_per_shift[s0, n1_] = []
                for n2 in instance.nurses_per_shift[s0]:
                    if n1_ != n2:
                        other_nurses_per_shift[s0, n1_].append(n2)
        self.other_nurses_per_shift = other_nurses_per_shift

    def execute(self):
        # initialize a solution
        self.initialize_solution()
        while self.cur_iter < self.max_iters:
            # if self.cur_iter % 1000 == 0:
            #     print(f"Iteration {self.cur_iter}")

            # Choose a movetype
            move_index = np.random.choice(4, p = self.move_prob_vector)
            move_type = self.move_options[move_index]

            # Apply the move
            if move_type == "MovePatient":
                self.do_MovePatient_move()
            elif move_type == "SwapPatients":
                self.do_SwapPatients_move()
            elif move_type == "ChangeNurse":
                self.do_ChangeNurse_move()
            else:
                move_available = self.do_RemoveNurse_move()
                if not move_available:
                    # distribute the probability of RemoveNurse across the other moves (according to ratios)
                    new_probabilities = self.move_prob_vector_if_fail

                    # reselect a move
                    move_index = np.random.choice(4, p = new_probabilities)
                    move_type = self.move_options[move_index]
                    if move_type == "MovePatient":
                        self.do_MovePatient_move()
                    elif move_type == "SwapPatients":
                        self.do_SwapPatients_move()
                    elif move_type == "ChangeNurse":
                        self.do_ChangeNurse_move()

            # update processes
            self.cur_obj_process.append(self.cur_obj)
            self.best_obj_process.append(self.best_obj)
            self.move_count[move_type] += 1

            # update temperature and iteration counter
            self.cur_temperature *= self.cooling_rate
            self.cur_iter += 1

    def do_MovePatient_move(self):
        """
        Will uniformly select a (non-occupant) patient p,
        then uniformly select a room r (different from current room).
        """

        # retrieve things for fast lookups
        instance = self.instance
        non_occupant_patients = self.instance.non_occupant_ids
        solution_attributes = self.solution_attributes

        p_index = np.random.choice(len(non_occupant_patients))
        p = non_occupant_patients[p_index]

        PR_assignment = self.cur_solution[0]
        r1 = PR_assignment[p]
        incomp_rooms = instance.patients[p]["incompatible_rooms"]

        room_candidates = []
        for r in instance.room_ids:
            if r != r1:
                if r not in incomp_rooms:
                    room_candidates.append(r)

        if len(room_candidates) == 0:
            raise Exception("No possible room to change to")

        r2_index = np.random.choice(len(room_candidates))
        r2 = room_candidates[r2_index]

        obj_delta, penalty_delta = delta_eval_MovePatient(instance, p, r1, r2, solution_attributes)
        obj_delta += penalty_delta

        # always accept the move if it improves, otherwise accept with a probability
        if obj_delta <= 0:
            accept_move = True
        else:
            acceptance_probability = np.exp(-obj_delta / self.cur_temperature)
            uniform_sample = np.random.uniform(0, 1)
            if uniform_sample < acceptance_probability:
                accept_move = True
            else:
                accept_move = False

        # accept_move = True
        if accept_move:
            # update the solution
            (PR_assignment, NR_assignment) = self.cur_solution
            PR_assignment[p] = r2
            self.cur_solution = (PR_assignment, NR_assignment)
            self.solution_attributes = update_solution_attributes_MovePatient(instance, p, r1, r2, self.cur_solution,
                                                                              solution_attributes)

            # update current and best objective
            self.cur_obj += obj_delta
            if self.cur_obj < self.best_obj:
                self.best_obj = self.cur_obj
                if self.store_best_solution: # improves speed when we only care about the objective
                    self.best_solution = copy_solution(self.cur_solution)
            # check if the objective is correct
            if self.do_objective_check:
                self.check_objective()

    def do_SwapPatients_move(self):
        """
        Will uniformly select a pair (p1,p2) where they overlap in shifts.
        The pair can only swap if their current room is compatible for the other and are in different rooms.
        The patients are all non-occupants
        """
        # retrieve things for fast lookup
        PR_assignment = self.cur_solution[0]
        instance = self.instance
        solution_attributes = self.solution_attributes

        # uniformly select a pair
        valid_patient_pairs = solution_attributes["valid_patient_pairs"]
        pair_index = np.random.choice(len(valid_patient_pairs))
        (p1,p2) = valid_patient_pairs[pair_index]
        r1 = PR_assignment[p1]
        r2 = PR_assignment[p2]

        # Evaluate
        obj_delta, penalty_delta = delta_eval_SwapPatients(instance, p1, p2, r1, r2,
                                                                         solution_attributes)
        obj_delta += penalty_delta

        # always accept the move if it improves, otherwise accept with a probability
        if obj_delta <= 0:
            accept_move = True
        else:
            acceptance_probability = np.exp(-obj_delta / self.cur_temperature)
            uniform_sample = np.random.uniform(0, 1)
            if uniform_sample < acceptance_probability:
                accept_move = True
            else:
                accept_move = False

        # accept_move = True
        if accept_move:
            # update the solution
            (PR_assignment, NR_assignment) = self.cur_solution
            PR_assignment[p1] = r2
            PR_assignment[p2] = r1
            self.cur_solution = (PR_assignment, NR_assignment)
            self.solution_attributes = update_solution_attributes_SwapPatients(instance, p1, p2, r1, r2,
                                                                               self.cur_solution,
                                                                               solution_attributes)

            # update current and best objective
            self.cur_obj += obj_delta
            if self.cur_obj < self.best_obj:
                self.best_obj = self.cur_obj
                if self.store_best_solution: # improves speed when we only care about the objective
                    self.best_solution = copy_solution(self.cur_solution)

            # check if the objective is correct
            if self.do_objective_check:
                self.check_objective()

    def do_ChangeNurse_move(self):
        """
        Will uniformly select a (r,s) pair where the nurse can be changed,
        then uniformly select another nurse that can be assigned to the room
        """

        # create local variables for fast lookup
        instance = self.instance
        solution_attributes = self.solution_attributes
        nurses_per_shift = instance.nurses_per_shift
        nurse_per_room = solution_attributes["nurse_per_room"]

        feasible_room_shift_pairs = self.feasible_room_shift_pairs

        index = np.random.choice(len(feasible_room_shift_pairs))
        (r,s) = feasible_room_shift_pairs[index]
        n1 = nurse_per_room[r,s]

        other_nurses_per_shift = self.other_nurses_per_shift
        other_nurses = other_nurses_per_shift[s,n1]

        n2_index = np.random.choice(len(other_nurses))
        n2 = other_nurses[n2_index]

        # Calculate the change in objective
        obj_delta = delta_eval_ChangeNurse(instance, r, s, n1, n2, solution_attributes)

        # always accept the move if it improves, otherwise accept with a probability
        if obj_delta <= 0:
            accept_move = True
        else:
            acceptance_probability = np.exp(-obj_delta / self.cur_temperature)
            uniform_sample = np.random.uniform(0, 1)
            if uniform_sample < acceptance_probability:
                accept_move = True
            else:
                accept_move = False

        # accept_move = True
        if accept_move:
            # update the solution
            (PR_assignment, NR_assignment) = self.cur_solution
            NR_assignment[n1, s].remove(r)
            NR_assignment[n2, s].append(r)
            self.cur_solution = (PR_assignment, NR_assignment)

            self.solution_attributes = update_solution_attributes_ChangeNurse(instance, r, s, n1, n2, solution_attributes)

            # update current and best objective
            self.cur_obj += obj_delta
            if self.cur_obj < self.best_obj:
                self.best_obj = self.cur_obj
                if self.store_best_solution: # improves speed when we only care about the objective
                    self.best_solution = copy_solution(self.cur_solution)
            # check if the objective is correct
            if self.do_objective_check:
                self.check_objective()

    def remove_nurse_helper(self):
        solution_attributes = self.solution_attributes
        instance = self.instance
        nurse_count_per_patient = solution_attributes["nurse_count_per_patient"]
        PR_assignment = self.cur_solution[0]
        NP_assignment = solution_attributes["NP_assignment"]

        RemoveNurse_options = solution_attributes["RemoveNurse_options"]
        patients_to_update = solution_attributes["patients_to_update"]

        for p in patients_to_update:
            r = PR_assignment[p]
            patient_shifts = instance.patients[p]["shifts"]
            patient_nurse_count = nurse_count_per_patient[p]
            options_for_p = []

            # for each assigned nurse, what happens if we cannot use that nurse anymore
            for n in patient_nurse_count:
                can_be_covered = True
                candidate_assignments = []
                for s in patient_shifts:
                    if NP_assignment[p, s] == n:  # if nurse assigned in shift
                        # is there another nurse that can cover the shift
                        other_nurse_available = []
                        for n0 in patient_nurse_count:
                            if n0 != n:
                                if n0 in instance.nurses_per_shift[s]:
                                    other_nurse_available.append(n0)
                        if len(other_nurse_available) == 0:
                            can_be_covered = False
                            break
                        else:
                            candidate_assignments.append((s, other_nurse_available))
                if can_be_covered and len(candidate_assignments) > 1:  # want to remove single changes
                    options_for_p.append((n,candidate_assignments))
            RemoveNurse_options[p] = options_for_p

        self.solution_attributes["RemoveNurse_options"] = RemoveNurse_options
        self.solution_attributes["patients_to_update"] = list()

        feasible_moves = []
        for p in RemoveNurse_options:
            r = PR_assignment[p]
            for (n,candidate_assignments) in RemoveNurse_options[p]:
                feasible_moves.append((p, r, n, candidate_assignments))
        return feasible_moves

    def do_RemoveNurse_move(self):
        """
        Will uniformly select a (p,n) pair to remove that nurse from consideration for that patient.
        For each shift that that nurse is used, select a random other nurse (in the set of nurses for p)
        """
        feasible_moves = self.remove_nurse_helper()

        if len(feasible_moves) == 0:
            return False

        # create local variables for fast lookup
        instance = self.instance
        solution_attributes = self.solution_attributes
        PR_assignment = self.cur_solution[0]

        # if len(feasible_moves) != len(self.temp_func1()):
        #     raise Exception(f"{len(feasible_moves)} != {len(self.temp_func1())}")

        # select a random (p,n) pair
        move_index = np.random.choice(len(feasible_moves))
        (p, r, n1, candidate_assignments) = feasible_moves[move_index]

        # For each shift, randomly select a nurse (from the nurse set of patient p)
        new_assignments = []
        for (s, nurses_available) in candidate_assignments:
            nurse_index = np.random.choice(len(nurses_available))
            n2 = nurses_available[nurse_index]
            new_assignments.append((s, n2))

        # evaluate this move
        obj_delta = delta_eval_RemoveNurse(instance, p, r, n1, new_assignments, PR_assignment, solution_attributes)

        # always accept the move if it improves, otherwise accept with a probability
        if obj_delta <= 0:
            accept_move = True
        else:
            acceptance_probability = np.exp(-obj_delta / self.cur_temperature)
            uniform_sample = np.random.uniform(0, 1)
            if uniform_sample < acceptance_probability:
                accept_move = True
            else:
                accept_move = False

        if accept_move:
            # update the solution
            (PR_assignment, NR_assignment) = self.cur_solution
            for (s, n2) in new_assignments:
                NR_assignment[n1, s].remove(r)
                NR_assignment[n2, s].append(r)
            self.cur_solution = (PR_assignment, NR_assignment)
            self.solution_attributes = update_solution_attributes_RemoveNurse(instance, p, r, n1, new_assignments,
                                                                              solution_attributes)

            # update current and best objective
            self.cur_obj += obj_delta
            if self.cur_obj < self.best_obj:
                self.best_obj = self.cur_obj
                if self.store_best_solution: # improves speed when we only care about the objective
                    self.best_solution = copy_solution(self.cur_solution)

            # check if the objective is correct
            if self.do_objective_check:
                self.check_objective()
        return True

    def initialize_solution(self):
        if self.initial_solution is None:
            if self.initial_solution_file_name is None:
                self.initial_solution = self.randomize_solution()
            else:
                self.initial_solution = self.retrieve_initial_solution()

        # get current solution, attributes and objective
        self.cur_solution = copy_solution(self.initial_solution)
        self.solution_attributes = get_SA_solution_attributes(self.instance, self.cur_solution)
        obj_value, penalty_value, obj_components, feasible = compute_objective_LS(self.instance,
                                                                                                 self.solution_attributes)
        self.cur_obj = obj_value + penalty_value

        # update best objective and solution
        self.best_obj = self.cur_obj
        if self.store_best_solution:
            self.best_solution = copy_solution(self.cur_solution)

        # Update processes
        self.cur_obj_process.append(self.cur_obj)
        self.best_obj_process.append(self.best_obj)

        # set temperature and cooling rate
        self.cur_temperature = self.T_start
        self.cooling_rate = (self.T_end / self.T_start) ** (1 / self.max_iters)

    def check_objective(self):
        # check if the objective matches the true objective
        solution_attributes = get_solution_attributes(self.instance, self.cur_solution)
        true_obj, true_penalty, _, _ = compute_objective_LS(self.instance, solution_attributes, False)
        true_obj += true_penalty

        if abs(true_obj - self.cur_obj) > 10e-6:
            raise Exception(f"Predicted objective {self.cur_obj} is different from true objective {true_obj}")

    def plot_process(self):
        # Show full process
        plt.plot(self.cur_obj_process, label="Current solution", alpha=0.8)
        plt.plot(self.best_obj_process, label="Best solution", alpha=0.8)
        plt.title(f"Optimization process\nInstance {self.instance.instance_name}")
        plt.show()

        # Show process zoomed in
        # mid_point_value = self.best_obj_process[len(self.best_obj_process) // 2]
        # y_min = self.best_obj_process[-1]
        # y_max = y_min + 3 * (mid_point_value - y_min)
        # y_diff = y_max - y_min
        # plt.plot(self.cur_obj_process, label="Current solution", alpha=0.8)
        # plt.plot(self.best_obj_process, label="Current solution", alpha=0.8)
        # plt.ylim([0.95*y_min, 1.55*y_max])
        # plt.title(f"Optimization process (zoomed in)\nInstance {self.instance.instance_name}")
        # plt.show()


    def retrieve_initial_solution(self):
        file_name = self.initial_solution_file_name
        file_path = self.initial_solution_path_name
        solution = pickle_load(file_name, file_path)
        return solution

    def randomize_solution(self):
        instance = self.instance
        PR_assignment = dict()
        for p in instance.occupant_ids:
            r = instance.patients[p]["prev_room"]
            PR_assignment[p] = r

        for p in instance.patient_ids:
            if p not in PR_assignment:
                comp_rooms = []
                for r in instance.room_ids:
                    if r not in instance.patients[p]["incompatible_rooms"]:
                        comp_rooms.append(r)
                r_index = np.random.choice(len(comp_rooms))
                r = comp_rooms[r_index]
                PR_assignment[p] = r

        # NR assignment
        NR_assignment = {(n,s): [] for n in instance.nurse_ids
                         for s in instance.nurses[n]["shifts"]}
        for r in instance.room_ids:
            for s in instance.all_shifts:
                nurses_present = instance.nurses_per_shift[s]
                nurse_index = np.random.choice(len(nurses_present))
                n = nurses_present[nurse_index]
                NR_assignment[n,s].append(r)

        solution = (PR_assignment, NR_assignment)
        return solution

    def reset_optimization(self):
        # Iterations
        self.cur_iter = 0

        # Solutions
        self.cur_solution = None
        self.best_solution = None

        # Solution attributes
        self.solution_attributes = None

        # processes
        self.cur_obj_process = []
        self.best_obj_process = []

        # Objectives
        self.cur_obj = float('inf')
        self.best_obj = float('inf')

if __name__ == "__main__":
    from InstanceClass import Instance

    name = "test03"
    file_name = name + ".json"
    dataset_name = instance_name_to_dataset(name)
    file_path = "Instances/" + dataset_name + "/" + file_name
    instance = Instance(file_path, print_instance_info=False)

    np.random.seed(42) # fix a seed
    simulated_annealer = SimulatedAnnealer(instance)

    simulated_annealer.store_best_solution = False # if False, does not store best solution to reduce computation time
    simulated_annealer.do_objective_check = False  # if True, checks if self.cur_obj is correct with self.cur_solution

    simulated_annealer.max_iters = 1000
    simulated_annealer.move_prob_vector = (0.2, 0.1, 0.6, 0.1)  # must sum to 1
    simulated_annealer.move_prob_vector_if_fail = (2 / 9, 1 / 9, 6 / 9, 0)

    start_time = datetime.datetime.now()
    simulated_annealer.execute()
    end_time = datetime.datetime.now()
    # simulated_annealer.plot_process()

    print(f"Total time taken = {end_time - start_time} = {(end_time - start_time).total_seconds()} seconds")
    print(f"Move frequency:")
    print(f"\tMovePatient: {simulated_annealer.move_count['MovePatient']} times = "
          f"{simulated_annealer.move_count['MovePatient'] / simulated_annealer.max_iters:.2f}%")
    print(f"\tSwapPatients: {simulated_annealer.move_count['SwapPatients']} times = "
          f"{simulated_annealer.move_count['SwapPatients'] / simulated_annealer.max_iters:.2f}%")
    print(f"\tChangeNurse: {simulated_annealer.move_count['ChangeNurse']} times = "
          f"{simulated_annealer.move_count['ChangeNurse'] / simulated_annealer.max_iters:.2f}%")
    print(f"\tRemoveNurse: {simulated_annealer.move_count['RemoveNurse']} times = "
          f"{simulated_annealer.move_count['RemoveNurse'] / simulated_annealer.max_iters:.2f}%")

    if simulated_annealer.store_best_solution:
        solution = simulated_annealer.best_solution
    else:
        solution = simulated_annealer.cur_solution
    solution_attributes = get_solution_attributes(instance, solution)
    obj_value, penalty_value, obj_components, feasible = compute_objective_LS(instance, solution_attributes, True)
    obj_value += penalty_value

    if abs(obj_value - simulated_annealer.best_obj) > 10e-6:
        print(f"WARNING! Values in this table ^ are not from the true best solution found")
        print(f"Objective value {obj_value:.3f} is different from best objective {simulated_annealer.best_obj:.3f}")
