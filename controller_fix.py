import re
with open('/home/danny/schproject/controller/garro_controller.py', 'r') as f:
    content = f.read()

# Replace set with dict
content = content.replace("self.flooded_srcs: set = set()", "self.flooded_srcs: dict = {}")

# Update packet_in_handler
old_handler_logic = """        if out_port == ofp.OFPP_FLOOD:
            flood_key = (dpid, hash(msg.data))
            if flood_key in self.flooded_srcs:
                # Already flooded this exact packet on this switch.
                # It's re-circulating through a mesh loop — drop it.
                return
            self.flooded_srcs.add(flood_key)"""

new_handler_logic = """        if out_port == ofp.OFPP_FLOOD:
            flood_key = (dpid, hash(msg.data))
            if flood_key in self.flooded_srcs:
                # Already flooded this exact packet on this switch.
                # It's re-circulating through a mesh loop — drop it.
                return
            import time
            self.flooded_srcs[flood_key] = time.time()"""
content = content.replace(old_handler_logic, new_handler_logic)

# Update _clear_flooded_srcs
old_clear = """    def _clear_flooded_srcs(self):
        \"\"\"Clear the broadcast flood-tracking set every 1 second.

        This breaks broadcast loops in the mesh topology (which take <100ms to recirculate)
        while allowing hosts to retry identical ARP requests (which typically retry after 1s).
        \"\"\"
        while True:
            hub.sleep(1)
            self.flooded_srcs.clear()"""

new_clear = """    def _clear_flooded_srcs(self):
        \"\"\"Clear the broadcast flood-tracking dict safely.

        Entries are kept for at least 1 second to ensure that propagating 
        broadcasts are fully suppressed across all mesh loops before being forgotten.
        \"\"\"
        import time
        while True:
            hub.sleep(1)
            now = time.time()
            stale = [k for k, v in self.flooded_srcs.items() if now - v > 1.0]
            for k in stale:
                del self.flooded_srcs[k]"""
content = content.replace(old_clear, new_clear)

with open('/home/danny/schproject/controller/garro_controller.py', 'w') as f:
    f.write(content)
print("Updated successfully")
