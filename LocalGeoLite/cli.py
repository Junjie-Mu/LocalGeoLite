import sys
import argparse
import signal
from typing import Optional
from . import load_model, unload_model, code, text

def signal_handler(sig, frame):
    print("\n👋 Exiting program...")
    unload_model()
    sys.exit(0)

def process_command(command: str) -> Optional[str]:
    command = command.strip()
    
    if command.lower() == "unload":
        return None
        
    if command.startswith("code "):
        prompt = command[5:].strip()
        if not prompt:
            return "❌ Please enter PROMPT"
        code(prompt)
        return None
        
    elif command.startswith("text "):
        prompt = command[5:].strip()
        if not prompt:
            return "❌ Please enter PROMPT"
        text(prompt)
        return None
        
    else:
        return ("❌ Invalid command. Please type 'code <prompt>' or 'text <prompt>' , or type 'unload' or press Ctrl+C "
                "to exit.")

def main():
    parser = argparse.ArgumentParser(description='LocalGeoLite CLI tool')
    parser.add_argument('command', choices=['loadmodel'], help='Startup command')
    parser.add_argument('cache_dir', nargs='?', help='Directory for model cache')
    
    args = parser.parse_args()
    
    if args.command == "loadmodel":
        print("🚀 Loading model...")
        load_model(cache_dir=args.cache_dir)
        print("✅ Model loaded successfully!")
        print("\n💡 Help:")
        print("- Type 'code <prompt>' to generate code.")
        print("- Type 'text <prompt>' to generate text.")
        print("- Type 'unload' or press Ctrl+C to exit.")
        print("\nWaiting for command...")

        signal.signal(signal.SIGINT, signal_handler)
        
        while True:
            try:
                command = input("\n🔍 >>> ")
                result = process_command(command)
                if result is None:
                    if command.lower() == "unload":
                        print("👋 Exiting program...")
                        unload_model()
                        break
                else:
                    print(result)
            except Exception as e:
                print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main() 