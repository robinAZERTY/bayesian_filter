"""open the .uml and use plantuml.preview command or shortcut to see the diagram"""

import os
import hpp2plantuml

folder2watch = os.path.join("include")
puml_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..","..", "docs","architecture/","classDiagram.puml")

hppFiles = [os.path.join(root, name) for root, dirs, files in os.walk(folder2watch) for name in files if name.endswith(".hpp")]
hpp2plantuml.CreatePlantUMLFile(hppFiles, puml_file_path)