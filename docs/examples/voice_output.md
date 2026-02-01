# Voice Output Examples

This document shows the transformation from raw RAG output to voice-optimized speech.

## Example 1: Power Supply Instructions

### RAG Output (Before)
```
Ensure the redundant power supply modules are configured in hot-swap mode and verify 
LED indicators according to Table 7-4. The PSU should display a solid green LED when 
functioning normally. If the LED shows amber, refer to the troubleshooting guide 
(Section 3.2.1) for detailed diagnostic procedures.
```

### Voice-Optimized Output (After)
```
Okay. First, make sure both power supplies are set to hot-swap mode.
You don't need to power off the system to replace them.
Now, check the L-E-D on each power supply.
A solid green light means it's working normally.
If you see an amber light, we'll need to troubleshoot.
Would you like me to walk you through that?
```

### Transformations Applied
- Removed "(Table 7-4)" reference
- Removed "(Section 3.2.1)" reference
- Split into 6 short sentences (avg 9 words each)
- "Ensure" → "Make sure"
- "PSU" → "power supply" / "power supplies"
- "LED" → "L-E-D" (spelled out)
- Added conversational marker "Okay"
- Added follow-up question

---

## Example 2: RAID Configuration

### RAG Output (Before)
```
Configure RAID5 on the PERC H755 controller using the lifecycle controller or BIOS 
configuration utility. Navigate to Device Settings > Controller Management > Create 
Virtual Disk. Select a minimum of 3 physical drives for RAID5. The stripe size 
should be set to 64KB for optimal performance with mixed workloads. Initialize the 
virtual disk before use (see Figure 5-12 for initialization progress).
```

### Voice-Optimized Output (After)
```
Let me walk you through setting up raid five.
First, go to your BIOS and find Device Settings.
Then select Controller Management.
Choose Create Virtual Disk.
You'll need at least three drives for raid five.
For the stripe size, use 64 kilobytes.
This works well for most workloads.
After creating the disk, you'll need to initialize it.
This might take a few minutes depending on the size.
```

### Transformations Applied
- "RAID5" → "raid five"
- "PERC H755" → simplified to "BIOS" context
- Menu path broken into steps
- Removed "(see Figure 5-12...)"
- "64KB" → "64 kilobytes"
- Added natural transitions ("Let me walk you through", "Then")
- Added helpful context ("This might take a few minutes")

---

## Example 3: Network Troubleshooting

### RAG Output (Before)
```
If the NIC LEDs are not illuminated, verify the following: (1) The network cable is 
properly seated in both the server NIC port and the switch; (2) The switch port is 
enabled and not configured for a different VLAN; (3) The SFP+ module (if applicable) 
is fully inserted and compatible with the 25GbE port specifications per IEEE 802.3by.
```

### Voice-Optimized Output (After)
```
If the network lights aren't on, let's check a few things.
First, make sure the cable is plugged in firmly.
Check both ends. The server side and the switch side.
Next, verify the switch port is turned on.
Also check that it's on the correct V-lan.
If you're using an S-F-P plus module, make sure it's pushed all the way in.
It needs to be compatible with 25 gigabit speeds.
Should I help you check any of these?
```

### Transformations Applied
- "NIC LEDs are not illuminated" → "network lights aren't on"
- Removed numbered list, used natural flow
- "SFP+" → "S-F-P plus"
- "25GbE" → "25 gigabit"
- Removed IEEE specification reference
- Added engagement question at end

---

## Example 4: Error Code Explanation

### RAG Output (Before)
```
Error code E2010 indicates a memory configuration error. This typically occurs when 
DIMM modules are installed in incorrect slots or when incompatible memory modules are 
detected. Refer to the memory population matrix (Table 4-2) for correct installation 
sequence. Ensure all DIMMs are DDR4 ECC Registered (RDIMM) type with matching speeds.
```

### Voice-Optimized Output (After)
```
Error E two zero one zero is a memory problem.
This usually means the memory sticks are in the wrong slots.
Or you might have mixed different types of memory.
Here's what to check.
Make sure all memory is the same type. D-D-R four E-C-C.
Also make sure they're in the right slots.
Start with slot A one, then A two, and so on.
Do you want me to explain the slot order?
```

### Transformations Applied
- "E2010" → "E two zero one zero"
- "DIMM modules" → "memory sticks"
- Removed "(Table 4-2)" reference
- "DDR4 ECC Registered (RDIMM)" → "D-D-R four E-C-C"
- Added simple slot order explanation
- Conversational engagement at end

---

## Voice Optimization Rules Summary

| Rule | Before | After |
|------|--------|-------|
| Max 15 words/sentence | Long compound sentences | Short, clear sentences |
| Remove references | "(see Table 7-4)" | *(removed)* |
| Spell acronyms | LED | L-E-D |
| Expand units | 64KB | 64 kilobytes |
| Simplify jargon | Ensure, utilize, module | Make sure, use, part |
| Expand RAID | RAID5 | raid five |
| Add markers | *(none)* | "Okay", "First", "Next" |
| End with engagement | *(statement)* | "Would you like...?" |
