import subprocess
import sys

def check_git():
    try:
        result = subprocess.run(["git", "status"], capture_output=True, text=True, cwd=r"c:\Users\admin\.openclaw\workspace\projects\travel_ir_system")
        print("=== GIT STATUS ===")
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        result2 = subprocess.run(["git", "remote", "-v"], capture_output=True, text=True, cwd=r"c:\Users\admin\.openclaw\workspace\projects\travel_ir_system")
        print("\n=== REMOTE ===")
        print(result2.stdout)
        
        result3 = subprocess.run(["git", "log", "--oneline", "-5"], capture_output=True, text=True, cwd=r"c:\Users\admin\.openclaw\workspace\projects\travel_ir_system")
        print("\n=== LAST 5 COMMITS ===")
        print(result3.stdout)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_git()