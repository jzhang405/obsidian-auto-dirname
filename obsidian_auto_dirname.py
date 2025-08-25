import os
import json
import click
from typing import List, Dict, Optional
from dotenv import load_dotenv
import openai
from openai import OpenAI
import re

# Load environment variables from .env file
load_dotenv()

class DirectoryManager:
    def __init__(self, input_dir: str = ".", locale: str = "en", keywords: Optional[List[str]] = None, 
                 method: str = "default", dry_run: bool = False, verbose: bool = False):
        self.input_dir = os.path.abspath(input_dir)
        self.locale = locale
        self.keywords = keywords or []
        self.method = method
        self.dry_run = dry_run
        self.verbose = verbose
        self.existing_dirs = self._get_existing_directories()
        self.generated_dirs = []
        
        # Initialize LLM client based on configuration
        self.llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
        self.llm_model = os.getenv("LLM_MODEL", "gpt-3.5-turbo-instruct")
        self.llm_client = self._initialize_llm_client()
    
    def _initialize_llm_client(self):
        """Initialize LLM client based on provider configuration"""
        api_key = os.getenv("LLM_API_KEY")
        base_url = os.getenv("LLM_BASE_URL")
        
        # Fallback to provider-specific variables if unified ones are not set
        if not api_key:
            api_key = os.getenv("LLM_API_KEY")
        
        if not base_url:
            base_url = os.getenv("LLM_BASE_URL")
        
        if not api_key:
            click.echo("Warning: LLM API key not found. Using placeholder responses.")
            return None
            
        if not base_url:
            click.echo("Warning: LLM base URL not found. Using default OpenAI URL.")
            base_url = "https://api.openai.com/v1"
            
        return OpenAI(api_key=api_key, base_url=base_url)
    
    def _get_existing_directories(self) -> List[str]:
        """Get list of existing directories in input path"""
        if not os.path.exists(self.input_dir):
            click.echo(f"Warning: Directory '{self.input_dir}' does not exist. Creating it.")
            if not self.dry_run:
                os.makedirs(self.input_dir, exist_ok=True)
            return []
        
        if not os.path.isdir(self.input_dir):
            click.echo(f"Error: '{self.input_dir}' is not a directory.")
            return []
            
        # Get all directories and filter out hidden directories
        all_dirs = [d for d in os.listdir(self.input_dir) if os.path.isdir(os.path.join(self.input_dir, d))]
        filtered_dirs = []
        
        for d in all_dirs:
            # Skip hidden directories like .obsidian, .trash, .git, etc.
            if d.startswith('.'):
                if self.verbose:
                    click.echo(f"Skipping hidden directory: {d}")
                continue
            
            # Skip specific directories
            if d in ['.obsidian', '.trash']:
                if self.verbose:
                    click.echo(f"Skipping directory: {d}")
                continue
                
            filtered_dirs.append(d)
            
        return filtered_dirs
    
    def _generate_directories_with_llm(self) -> List[Dict]:
        """Use LLM to generate directory structure based on keywords and method"""
        if not self.keywords:
            # Generate default directories when no keywords provided
            return self._generate_default_directories()
        
        # Generate directories based on keywords using LLM
        return self._generate_from_keywords_with_llm()
    
    def _process_llm_response(self, response: str) -> List[Dict]:
        """Process LLM response and convert to directory structure"""
        # Try to parse LLM response as JSON
        if self.llm_client and response and not response.startswith("Placeholder"):
            try:
                # Clean up the response to extract JSON
                # Remove markdown code block markers if present
                response = response.replace("```json", "").replace("```", "").strip()
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    parsed_response = json.loads(json_str)
                    # Validate the structure
                    if isinstance(parsed_response, list) and len(parsed_response) > 0:
                        # Group items by top-level directory
                        top_level_dirs = {}
                        
                        for item in parsed_response:
                            if isinstance(item, dict) and "path" in item and "subcategories" in item:
                                path = item["path"]
                                subcategories = item["subcategories"]
                                
                                # Process subcategories to ensure they are simple names
                                processed_subcategories = []
                                for sub in subcategories:
                                    # If subcategory contains path separators, take only the last part
                                    if "/" in sub or "\\\\" in sub:
                                        sub = sub.split("/")[-1].split("\\\\")[-1]
                                    processed_subcategories.append(sub)
                                
                                # Check if this is a nested path (contains "/")
                                if "/" in path:
                                    # Extract top-level directory name
                                    top_level_dir = path.split("/")[0]
                                    
                                    # Add subcategories to the top-level directory
                                    if top_level_dir not in top_level_dirs:
                                        top_level_dirs[top_level_dir] = {
                                            "path": top_level_dir,
                                            "subcategories": []
                                        }
                                    top_level_dirs[top_level_dir]["subcategories"].extend(processed_subcategories)
                                else:
                                    # This is a top-level directory
                                    if path not in top_level_dirs:
                                        top_level_dirs[path] = {
                                            "path": path,
                                            "subcategories": []
                                        }
                                    top_level_dirs[path]["subcategories"].extend(processed_subcategories)
                        
                        # Convert to list
                        return list(top_level_dirs.values())
            except Exception as e:
                if self.verbose:
                    click.echo(f"Error parsing LLM response as JSON: {e}")
        
        # Fallback to placeholder if no API key is available or parsing failed
        dirs = []
        for i, keyword in enumerate(self.keywords, 1):
            dirs.append({
                "path": f"{i:02d} - {keyword.capitalize()}",
                "subcategories": [f"Sub-{keyword}-1", f"Sub-{keyword}-2", f"Sub-{keyword}-3"]
            })
        return dirs
    
    def _generate_default_directories(self) -> List[Dict]:
        """Generate default directory structure"""
        # Use LLM to generate default directories
        prompt = f"Generate a default Obsidian directory structure for {self.locale} locale with 2-3 levels. Include common categories like work, personal, etc. "
        prompt += "Format the response as a JSON array of objects with 'path' and 'subcategories' keys. "
        prompt += "For main directories, follow the existing naming convention in the vault. If the existing directories use the format '01 - Directory Name', use that format. Otherwise, use '01. Directory Name'. "
        prompt += "For subcategories, use simple names without path prefixes. "
        prompt += "Return only the JSON array, nothing else. Example format: "
        prompt += '[{"path": "01 - Work", "subcategories": ["Projects", "Meetings", "Documents"]}]'
        
        # Add information about existing directories to the prompt
        if self.existing_dirs:
            prompt += f"\n\nThe vault already contains the following directories: {', '.join(self.existing_dirs)}. "
            prompt += "Please try to integrate the new structure with these existing directories, avoiding duplication. "
            prompt += "First, use existing directories where appropriate. Only create new directories if necessary. "
            prompt += "If an existing directory doesn't have subcategories, you can add them. "
            prompt += "If an existing directory already has subcategories, preserve them and only add new ones if needed."
        
        response = self._call_llm(prompt)
        
        # Print prompt and response for debugging if verbose mode is enabled
        if self.verbose:
            click.echo(f"Prompt: {prompt}")
            click.echo(f"Response: {response}")
        
        return self._process_llm_response(response)
    
    def _generate_from_keywords_with_llm(self) -> List[Dict]:
        """Generate directory structure from keywords using LLM"""
        prompt = f"You are an expert in organizing knowledge. Generate an Obsidian directory structure for {self.locale} locale based on these keywords: {', '.join(self.keywords)}. "
        prompt += f"Use the {self.method} organizational method. Create 2-3 levels of directories with 3-5 subcategories each. "
        prompt += "Format the response STRICTLY as a JSON array of objects with 'path' and 'subcategories' keys. "
        prompt += "For main directories, follow the existing naming convention in the vault. If the existing directories use the format '01 - Directory Name', use that format. Otherwise, use '01. Directory Name'. "
        prompt += "For subcategories, use simple names without path prefixes. "
        prompt += "Return ONLY the JSON array with no additional text, no markdown, no explanations. "
        prompt += "Example format: "
        prompt += '[{"path": "01 - Programming", "subcategories": ["Languages", "Concepts", "Tools"]}]'
        
        # Add information about existing directories to the prompt
        if self.existing_dirs:
            prompt += f"\n\nThe vault already contains the following directories: {', '.join(self.existing_dirs)}. "
            prompt += "Please try to integrate the new structure with these existing directories, avoiding duplication. "
            prompt += "First, use existing directories where appropriate. Only create new directories if necessary. "
            prompt += "If an existing directory doesn't have subcategories, you can add them. "
            prompt += "If an existing directory already has subcategories, preserve them and only add new ones if needed."
        
        response = self._call_llm(prompt)
        
        # Print prompt and response for debugging if verbose mode is enabled
        if self.verbose:
            click.echo(f"Prompt: {prompt}")
            click.echo(f"Response: {response}")
        
        return self._process_llm_response(response)
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM API with prompt"""
        if not self.llm_client:
            # Return a placeholder response if no API key is available
            return "Placeholder LLM response"
        
        if self.verbose:
            click.echo(f"Calling LLM with model: {self.llm_model}")
        
        try:
            # Call LLM API
            response = self.llm_client.completions.create(
                model=self.llm_model,
                prompt=prompt,
                max_tokens=int(os.getenv("LLM_MAX_TOKENS", "500")),
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.7"))
            )
            result = response.choices[0].text.strip()
            if self.verbose:
                click.echo(f"LLM response received: {result}")
            return result
        except Exception as e:
            click.echo(f"Error calling LLM API: {e}")
            return "Placeholder LLM response"
    
    def _format_directory_name(self, name: str, priority: int) -> str:
        """Format directory name with priority prefix"""
        return f"{priority:02d} - {name}"
    
    def _is_new_directory(self, path: str) -> bool:
        """Check if a directory is new (not existing before)"""
        normalized_path = self._normalize_directory_name(path)
        for existing_dir in self.existing_dirs:
            normalized_existing = self._normalize_directory_name(existing_dir)
            if normalized_path == normalized_existing or normalized_path in normalized_existing or normalized_existing in normalized_path:
                return False
        return True
    
    def _create_directory_tree_preview(self, dirs: List[Dict]) -> str:
        """Create a tree preview of the directory structure"""
        preview = f"Directory Structure Preview for '{self.input_dir}':\n"
        
        # Track which directories we've already shown to avoid duplicates
        shown_dirs = set()
        
        for dir_info in dirs:
            path = dir_info['path']
            
            # Skip if we've already shown this directory
            if path in shown_dirs:
                continue
                
            shown_dirs.add(path)
            
            # Green for new directories, use default color for existing ones
            if self._is_new_directory(path):
                preview += f"\033[32m├── {path} (new)\033[0m\n"
            else:
                preview += f"├── {path} (existing)\n"
                
            # Show subcategories only if they exist
            subcategories = dir_info.get('subcategories', [])
            if subcategories:
                for subcategory in subcategories:
                    # Check if subcategory is new
                    is_new_sub = True  # By default, assume new
                    # In a more sophisticated implementation, we could check actual filesystem
                    if is_new_sub:
                        preview += f"\033[32m│   └── {subcategory} (new)\033[0m\n"
                    else:
                        preview += f"│   └── {subcategory} (existing)\n"
            else:
                # For existing directories with no subcategories, show a placeholder
                if not self._is_new_directory(path):
                    preview += f"│   └── (no subcategories generated)\n"
                    
        return preview
    
    def _normalize_directory_name(self, name: str) -> str:
        """Normalize directory name by removing priority prefix if exists"""
        # Remove priority prefix like "01." if exists
        if "." in name and name.split(".")[0].isdigit():
            return name.split(".", 1)[1].strip()
        
        # Remove priority prefix like "01 - " if exists
        if " - " in name and name.split(" - ")[0].isdigit():
            return name.split(" - ", 1)[1].strip()
            
        return name.strip()
    
    def _find_matching_existing_dir(self, generated_name: str, existing_dirs: List[str]) -> Optional[str]:
        """Find matching existing directory for a generated name"""
        normalized_generated = self._normalize_directory_name(generated_name).lower()
        
        for existing_dir in existing_dirs:
            normalized_existing = self._normalize_directory_name(existing_dir).lower()
            # Direct match
            if normalized_existing == normalized_generated:
                return existing_dir
            # Partial match (generated is subset of existing)
            if normalized_generated in normalized_existing:
                return existing_dir
            # Partial match (existing is subset of generated)
            if normalized_existing in normalized_generated:
                return existing_dir
            # Special case: handle "学习" matching "30 - 学习"
            if normalized_generated.replace(" ", "") in normalized_existing.replace(" ", ""):
                return existing_dir
            if normalized_existing.replace(" ", "") in normalized_generated.replace(" ", ""):
                return existing_dir
        return None
    
    def _merge_with_existing(self, generated_dirs: List[Dict]) -> List[Dict]:
        """Merge generated directories with existing ones"""
        if not self.existing_dirs:
            return generated_dirs
            
        # Create a result list that preserves existing directories
        merged_dirs = []
        used_existing_dirs = set()
        
        # First, add all existing directories with their current subcategories
        for existing_dir in self.existing_dirs:
            # Try to find if this existing directory has generated subcategories
            matching_gen_dir = None
            for gen_dir in generated_dirs:
                gen_path = gen_dir["path"]
                # Handle both simple and nested paths
                if "/" in gen_path:
                    top_level_dir = gen_path.split("/")[0]
                    if self._find_matching_existing_dir(top_level_dir, [existing_dir]):
                        matching_gen_dir = gen_dir
                        break
                else:
                    if self._find_matching_existing_dir(gen_path, [existing_dir]):
                        matching_gen_dir = gen_dir
                        break
            
            # Add existing directory with its current structure
            if matching_gen_dir:
                # Merge with generated subcategories
                existing_subcategories = []  # In practice, we don't read existing subdirs from filesystem
                generated_subcategories = matching_gen_dir.get("subcategories", [])
                # Combine and deduplicate
                all_subcategories = list(set(existing_subcategories + generated_subcategories))
                merged_dirs.append({
                    "path": existing_dir,
                    "subcategories": all_subcategories
                })
                used_existing_dirs.add(existing_dir)
            else:
                # No generated subcategories for this existing directory
                merged_dirs.append({
                    "path": existing_dir,
                    "subcategories": []
                })
        
        # Add new directories that don't match existing ones
        for gen_dir in generated_dirs:
            path = gen_dir["path"]
            
            # Check if this is a nested path (contains "/")
            if "/" in path:
                # Extract top-level directory name
                top_level_dir = path.split("/")[0]
                
                # Check if this top-level directory matches an existing one
                matching_existing = self._find_matching_existing_dir(top_level_dir, self.existing_dirs)
                
                if not matching_existing:
                    # This is a new top-level directory
                    merged_dirs.append(gen_dir)
            else:
                # This is a top-level directory
                matching_existing = self._find_matching_existing_dir(path, self.existing_dirs)
                
                if not matching_existing:
                    # This is a new directory
                    merged_dirs.append(gen_dir)
        
        return merged_dirs
    
    def _save_to_file(self, dirs: List[Dict]):
        """Save directory structure to a JSON file"""
        data = {"categories": dirs}
        output_file = os.path.join(self.input_dir, "directory_structure.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        if self.verbose:
            click.echo(f"Directory structure saved to JSON file: {output_file}")
    
    def generate(self):
        """Main method to generate directory structure"""
        # Generate directories using LLM
        generated_dirs = self._generate_directories_with_llm()
        
        # Merge with existing directories
        merged_dirs = self._merge_with_existing(generated_dirs)
        
        # Create preview
        preview = self._create_directory_tree_preview(merged_dirs)
        click.echo(preview)
        
        # Create directories and save to file if not in dry-run mode
        if not self.dry_run:
            # Create the actual directory structure
            self._create_directories(merged_dirs)
            
            # Save to file
            self._save_to_file(merged_dirs)
            output_file = os.path.join(self.input_dir, "directory_structure.json")
            click.echo(f"Directory structure saved to {output_file}")
            
            # Create Uncategorized directory if needed
            uncategorized_dir = os.path.join(self.input_dir, "Uncategorized")
            if not os.path.exists(uncategorized_dir):
                os.makedirs(uncategorized_dir)
                click.echo(f"Created Uncategorized directory: {uncategorized_dir}")
        else:
            click.echo("Dry-run mode: No directories were created")
    
    def _create_directories(self, dirs: List[Dict]):
        """Create the actual directory structure"""
        for dir_info in dirs:
            # Create main directory
            main_dir_path = os.path.join(self.input_dir, dir_info["path"])
            if not os.path.exists(main_dir_path):
                os.makedirs(main_dir_path)
                if self.verbose:
                    click.echo(f"Created directory: {main_dir_path}")
            
            # Create subcategories
            for subcategory in dir_info.get("subcategories", []):
                # Handle nested paths (like "02. 领域/01.个人成长")
                if "/" in subcategory:
                    # For nested paths, create the full path
                    sub_path = os.path.join(self.input_dir, dir_info["path"], subcategory)
                    os.makedirs(sub_path, exist_ok=True)
                    if self.verbose:
                        click.echo(f"Created directory: {sub_path}")
                else:
                    # Simple subcategory
                    sub_dir_path = os.path.join(self.input_dir, dir_info["path"], subcategory)
                    if not os.path.exists(sub_dir_path):
                        os.makedirs(sub_dir_path)
                        if self.verbose:
                            click.echo(f"Created directory: {sub_dir_path}")


@click.command()
@click.option('--input-dir', default='.', help='Input directory path (default: current directory)')
@click.option('--locale', default='en', help='Locale for directory names (default: en)')
@click.option('--keywords', help='Comma-separated list of keywords to use as seeds')
@click.option('--method', type=click.Choice(['PARA', 'Zettelkasten', 'default']), 
              default='default', help='Organizational method')
@click.option('--dry-run', is_flag=True, help='Preview mode without generating directories')
@click.option('--verbose', is_flag=True, help='Enable verbose mode to print prompts and responses')
def main(input_dir: str, locale: str, keywords: str, method: str, dry_run: bool, verbose: bool):
    """Obsidian Auto Directory Name - Automatically generate Obsidian directory structures"""
    keyword_list = [k.strip() for k in keywords.split(',')] if keywords else None
    
    manager = DirectoryManager(
        input_dir=input_dir,
        locale=locale,
        keywords=keyword_list,
        method=method,
        dry_run=dry_run,
        verbose=verbose
    )
    
    manager.generate()


if __name__ == '__main__':
    main()