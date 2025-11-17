import sys
sys.path.insert(0, 'E:/Agentic_Wrapper')

print("Testing Agents import...")
try:
    import Agents
    print(f"Agents type: {type(Agents)}")
    print(f"Agents attributes: {dir(Agents)}")
    
    from Agents import Agents as AgentsClass
    print(f"AgentsClass type: {type(AgentsClass)}")
    print("✓ Import successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Testing Tasks import...")
try:
    import Tasks
    print(f"Tasks type: {type(Tasks)}")
    print(f"Tasks attributes: {dir(Tasks)}")
    
    from Tasks import Tasks as TasksClass
    print(f"TasksClass type: {type(TasksClass)}")
    print("✓ Import successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()