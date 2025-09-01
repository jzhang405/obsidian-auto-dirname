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
                 method: str = "default", min_dirs: int = 5, max_dirs: int = 10, 
                 dry_run: bool = False, verbose: bool = False):
        self.input_dir = os.path.abspath(input_dir)
        self.locale = locale
        self.keywords = keywords or []
        self.method = method
        self.min_dirs = min_dirs
        self.max_dirs = max_dirs
        self.dry_run = dry_run
        self.verbose = verbose
        self.existing_dirs = self._get_existing_directories()
        self.generated_dirs = []
        
        # Initialize LLM client based on configuration
        self.llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
        self.llm_model = os.getenv("LLM_MODEL", "gpt-3.5-turbo-instruct")
        self.llm_client = self._initialize_llm_client()
    
    def _initialize_llm_client(self):
        """Initialize LLM client using Unified LLM Configuration"""
        # Use unified configuration (recommended)
        api_key = os.getenv("LLM_API_KEY")
        base_url = os.getenv("LLM_BASE_URL")
        
        if self.verbose:
            click.echo(f"Using unified LLM configuration:")
            click.echo(f"  Provider: {self.llm_provider}")
            click.echo(f"  Model: {self.llm_model}")
            click.echo(f"  Base URL: {base_url}")
            click.echo(f"  API Key: {'***' + api_key[-4:] if api_key else 'Not configured'}")
        
        # Fallback to provider-specific variables only if unified ones are not set
        if not api_key:
            if self.verbose:
                click.echo("Unified LLM_API_KEY not found, trying provider-specific keys...")
            if self.llm_provider == "deepseek":
                api_key = os.getenv("DEEPSEEK_API_KEY")
            elif self.llm_provider == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
            elif self.llm_provider == "custom":
                api_key = os.getenv("CUSTOM_LLM_API_KEY")
        
        if not base_url:
            if self.verbose:
                click.echo("Unified LLM_BASE_URL not found, trying provider-specific URLs...")
            if self.llm_provider == "deepseek":
                base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
            elif self.llm_provider == "openai":
                base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            elif self.llm_provider == "custom":
                base_url = os.getenv("CUSTOM_LLM_BASE_URL")
            else:
                base_url = "https://api.openai.com/v1"
        
        if not api_key:
            click.echo("Error: LLM API key not found in unified or provider-specific configuration.")
            click.echo("Please set LLM_API_KEY in your .env file or the appropriate provider-specific key.")
            return None
            
        if not base_url:
            click.echo("Error: LLM base URL not found in unified or provider-specific configuration.")
            click.echo("Please set LLM_BASE_URL in your .env file.")
            return None
            
        if self.verbose:
            click.echo(f"✓ Successfully configured LLM client with {self.llm_provider} provider")
            
        return OpenAI(api_key=api_key, base_url=base_url)
    
    def _get_existing_directories(self) -> List[str]:
        """Get list of existing directories in input path that match naming convention"""
        if not os.path.exists(self.input_dir):
            click.echo(f"Warning: Directory '{self.input_dir}' does not exist. Creating it.")
            if not self.dry_run:
                os.makedirs(self.input_dir, exist_ok=True)
            return []
        
        if not os.path.isdir(self.input_dir):
            click.echo(f"Error: '{self.input_dir}' is not a directory.")
            return []
            
        # Get all directories and filter based on naming convention
        all_dirs = [d for d in os.listdir(self.input_dir) if os.path.isdir(os.path.join(self.input_dir, d))]
        filtered_dirs = []
        
        for d in all_dirs:
            # Skip hidden directories like .obsidian, .trash, .git, etc.
            if d.startswith('.'):
                if self.verbose:
                    click.echo(f"Skipping hidden directory: {d}")
                continue
            
            # Only include directories that match the naming convention: 二位数字 + 占位符 + 目录名
            # Format: "01.DirectoryName" or "01. DirectoryName" or "01 - DirectoryName"
            if (re.match(r'^\d{2}\.\w+', d) or  # 01.Work
                re.match(r'^\d{2}\. \w+', d) or  # 01. Work  
                re.match(r'^\d{2} - \w+', d)):   # 01 - Work
                filtered_dirs.append(d)
            else:
                if self.verbose:
                    click.echo(f"Ignoring directory (doesn't match naming convention): {d}")
                    
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
                        
                        # Convert to list and apply directory count limits
                        result_dirs = list(top_level_dirs.values())
                        return self._apply_directory_limits(result_dirs)
            except Exception as e:
                if self.verbose:
                    click.echo(f"Error parsing LLM response as JSON: {e}")
        
        # Fallback to placeholder if no API key is available or parsing failed
        dirs = []
        # Use keywords if available, otherwise generate default categories
        if self.keywords:
            # Limit to max_dirs number of keywords
            limited_keywords = self.keywords[:self.max_dirs]
            # Ensure we have at least min_dirs keywords
            while len(limited_keywords) < self.min_dirs and len(limited_keywords) < len(self.keywords):
                limited_keywords = self.keywords[:len(limited_keywords) + 1]
            
            for i, keyword in enumerate(limited_keywords, 1):
                dirs.append({
                    "path": f"{i:02d}.{keyword.capitalize()}",
                    "subcategories": [f"Sub-{keyword}-1", f"Sub-{keyword}-2", f"Sub-{keyword}-3"]
                })
        else:
            # Generate default placeholder categories
            default_categories = ["Work", "Personal", "Study", "Projects", "Resources", "Archive", "Notes", "Ideas", "Health", "Finance"]
            # Select appropriate number of categories based on min/max constraints
            num_dirs = min(max(self.min_dirs, 5), self.max_dirs)
            selected_categories = default_categories[:num_dirs]
            
            for i, category in enumerate(selected_categories, 1):
                dirs.append({
                    "path": f"{i:02d}.{category}",
                    "subcategories": [f"Sub-{category.lower()}-1", f"Sub-{category.lower()}-2", f"Sub-{category.lower()}-3"]
                })
        return dirs
    
    def _apply_directory_limits(self, dirs: List[Dict]) -> List[Dict]:
        """Apply min/max directory count limits to the generated directories"""
        if len(dirs) < self.min_dirs:
            if self.verbose:
                click.echo(f"Warning: Generated {len(dirs)} directories, but minimum is {self.min_dirs}. The response may need more directories.")
        elif len(dirs) > self.max_dirs:
            if self.verbose:
                click.echo(f"Limiting directories from {len(dirs)} to {self.max_dirs} (maximum allowed)")
            dirs = dirs[:self.max_dirs]
        
        return dirs
    
    def _generate_default_directories(self) -> List[Dict]:
        """Generate default directory structure"""
        # Use LLM to generate default directories - strictly 2 levels only
        prompt = f"Generate a default Obsidian directory structure for {self.locale} locale with exactly 2 levels. Include common categories like work, personal, etc. "
        prompt += f"Generate between {self.min_dirs} and {self.max_dirs} main directories. "
        prompt += "Format the response as a JSON array of objects with 'path' and 'subcategories' keys. "
        prompt += "For main directories, use the format '01.DirectoryName' (number + dot + name, NO SPACE after dot). "
        prompt += "For subcategories, use simple names without path prefixes or numbers. "
        prompt += "Return only the JSON array, nothing else. Example format: "
        prompt += '[{"path": "01.Work", "subcategories": ["Projects", "Meetings", "Documents"]}]'
        
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
        # Check if PARA method should use 3-level structure (conflict with 2-level rule)
        if self.method == "PARA":
            prompt = f"You are an expert in organizing knowledge using PARA method. Generate an Obsidian directory structure for {self.locale} locale based on these keywords: {', '.join(self.keywords)}. "
            prompt += f"Use PARA organizational method with exactly 3 levels. Generate between {self.min_dirs} and {self.max_dirs} main directories based on the keywords provided. "
            prompt += "Format the response STRICTLY as a JSON array of objects with 'path' and 'subcategories' keys. "
            prompt += "For main directories, use the format '01.DirectoryName' (number + dot + name, NO SPACE after dot). "
            prompt += "For subcategories, use simple names without path prefixes or numbers. "
            prompt += "Return ONLY the JSON array with no additional text, no markdown, no explanations. "
            prompt += "Example format: "
            prompt += '[{"path": "01.Projects", "subcategories": ["Active", "Planning", "Archive"]}]'
        else:
            prompt = f"You are an expert in organizing knowledge. Generate an Obsidian directory structure for {self.locale} locale based on these keywords: {', '.join(self.keywords)}. "
            prompt += f"Use the {self.method} organizational method. Create exactly 2 levels of directories with 3-5 subcategories each. "
            prompt += f"Generate between {self.min_dirs} and {self.max_dirs} main directories based on the keywords provided. "
            prompt += "Format the response STRICTLY as a JSON array of objects with 'path' and 'subcategories' keys. "
            prompt += "For main directories, use the format '01.DirectoryName' (number + dot + name, NO SPACE after dot). "
            prompt += "For subcategories, use simple names without path prefixes or numbers. "
            prompt += "Return ONLY the JSON array with no additional text, no markdown, no explanations. "
            prompt += "Example format: "
            prompt += '[{"path": "01.Programming", "subcategories": ["Languages", "Concepts", "Tools"]}]'
        
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
        """Call LLM API with prompt using modern chat completion API"""
        if not self.llm_client:
            # Return a placeholder response if no API key is available
            return "Placeholder LLM response"
        
        if self.verbose:
            click.echo(f"Calling LLM with provider: {self.llm_provider}, model: {self.llm_model}")
        
        try:
            # Use modern chat completion API instead of legacy completion API
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are an expert in organizing knowledge and creating directory structures for Obsidian vaults. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=int(os.getenv("LLM_MAX_TOKENS", "500")),
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.7"))
            )
            result = response.choices[0].message.content.strip()
            if self.verbose:
                click.echo(f"LLM response received: {result}")
            return result
        except Exception as e:
            click.echo(f"Error calling LLM API: {e}")
            if self.verbose:
                click.echo(f"Falling back to placeholder response")
            return "Placeholder LLM response"
    
    def _format_directory_name(self, name: str, priority: int) -> str:
        """Format directory name with priority prefix"""
        return f"{priority:02d}.{name}"
    
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
        # Remove priority prefix like "01." if exists (new format without space)
        if "." in name and len(name.split(".")[0]) == 2 and name.split(".")[0].isdigit():
            return name.split(".", 1)[1].strip()
        
        # Remove priority prefix like "01. " if exists (old format with space for backward compatibility)
        if ". " in name and name.split(". ")[0].isdigit():
            return name.split(". ", 1)[1].strip()
        
        # Remove priority prefix like "01 - " if exists (legacy format for backward compatibility)
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
        
        # Add "未分类" (Uncategorized) directory according to spec requirement
        self._add_uncategorized_directory(merged_dirs)
        
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
        else:
            click.echo("Dry-run mode: No directories were created")
    
    def _add_uncategorized_directory(self, dirs: List[Dict]):
        """Add 未分类 (Uncategorized) directory according to spec requirement"""
        # Determine the next priority number
        existing_numbers = []
        for dir_info in dirs:
            path = dir_info['path']
            # Extract number from path like "01.Work" or "01. Work" or "01 - Work"
            if re.match(r'^\d{2}', path):
                num_str = path[:2]
                if num_str.isdigit():
                    existing_numbers.append(int(num_str))
        
        next_num = max(existing_numbers, default=0) + 1
        
        # Generate uncategorized directory name based on locale
        if self.locale == 'zh':
            uncategorized_name = "未分类"
        else:
            uncategorized_name = "Uncategorized"
        
        # Add uncategorized directory
        uncategorized_dir = {
            "path": f"{next_num:02d}.{uncategorized_name}",
            "subcategories": []
        }
        
        dirs.append(uncategorized_dir)
        
        if self.verbose:
            click.echo(f"Added {uncategorized_name} directory: {uncategorized_dir['path']}")
    
    def _create_directories(self, dirs: List[Dict]):
        """Create the actual directory structure - strictly 2 levels only"""
        for dir_info in dirs:
            # Create main directory
            main_dir_path = os.path.join(self.input_dir, dir_info["path"])
            if not os.path.exists(main_dir_path):
                os.makedirs(main_dir_path)
                if self.verbose:
                    click.echo(f"Created directory: {main_dir_path}")
            
            # Create subcategories - only 2 levels, no nesting allowed
            for subcategory in dir_info.get("subcategories", []):
                # Ignore any nested paths - flatten to simple subcategory names
                if "/" in subcategory:
                    # Take only the last part if there's a nested path
                    subcategory = subcategory.split("/")[-1]
                
                # Simple subcategory - second level only
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
@click.option('--min', 'min_dirs', default=5, help='Minimum number of directories to generate (default: 5)')
@click.option('--max', 'max_dirs', default=10, help='Maximum number of directories to generate (default: 10)')
@click.option('--dry-run', is_flag=True, help='Preview mode without generating directories')
@click.option('--verbose', is_flag=True, help='Enable verbose mode to print prompts and responses')
def main(input_dir: str, locale: str, keywords: str, method: str, min_dirs: int, max_dirs: int, dry_run: bool, verbose: bool):
    """Obsidian Auto Directory Name - Automatically generate Obsidian directory structures"""
    # Validate min/max parameters
    if min_dirs < 1:
        click.echo("Error: Minimum number of directories must be at least 1")
        return
    if max_dirs < min_dirs:
        click.echo("Error: Maximum number of directories must be greater than or equal to minimum")
        return
    if max_dirs > 20:
        click.echo("Warning: Maximum number of directories is very large, consider reducing it for better organization")
    
    keyword_list = [k.strip() for k in keywords.split(',')] if keywords else None
    
    manager = DirectoryManager(
        input_dir=input_dir,
        locale=locale,
        keywords=keyword_list,
        method=method,
        min_dirs=min_dirs,
        max_dirs=max_dirs,
        dry_run=dry_run,
        verbose=verbose
    )
    
    manager.generate()


if __name__ == '__main__':
    main()