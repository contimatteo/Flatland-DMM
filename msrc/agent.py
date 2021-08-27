
class TreeLookupAgent:
    def act(self, handle, obs, info):
        if obs is None or obs[handle] is None:  # Safeguard
            return 0

        if not info['action_required'][handle]:  # Act only if required
            return 0

        root = obs[handle]._asdict()

        # Get the best direction, based on the 1st ply of the tree observation
        MAX_DIST = 100000
        min_dist, best_dir = MAX_DIST, None
        childs = root['childs']
        for direction in ['L', 'F', 'R', 'B']:
            dir_node = childs.get(direction, {})
            if isinstance(dir_node, float):  # Check if -inf (no node)
                continue
            dir_dist = dir_node.dist_min_to_target
            if min_dist > dir_dist:
                min_dist = dir_dist
                best_dir = direction

        # Get the numeric action from the direction
        action = 1
        if best_dir == 'L':
            action = 1
        elif best_dir == 'F':
            action = 2
        elif best_dir == 'R':
            action = 3

        print(f"Actor {handle} performed {best_dir} with distance {min_dist}")
        return action