# -*- coding: utf-8 -*-
"""Script Maestro - Ejecución Completa"""
import os
import sys

# 📜 Lista de scripts a ejecutar
scripts = [
    "CreditosActivos10.py",
    "Rentabilidad.py",
    "Detallado de Canje.py",
    "Kpis's.py"
]

print("🚀 Iniciando ejecución de scripts...\n")

# ▶️ Ejecutar cada script
for script in scripts:
    print(f"{'='*60}")
    print(f"🔄 Ejecutando: {script}")
    print(f"{'='*60}")
    
    try:
        with open(script, "r", encoding="utf-8") as file:
            exec(file.read(), {'__name__': '__main__'})
        print(f"✅ {script} completado exitosamente.\n")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo {script}\n")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error al ejecutar {script}:")
        print(f"   {str(e)}\n")
        sys.exit(1)

print(f"{'='*60}")
print("🎯 ¡Todas las tareas completadas exitosamente!")
print(f"{'='*60}")
```

### 5. Estructura final de tu repositorio
```
tu-repo/
├── .github/
│   └── workflows/
│       └── ejecucion_completa.yml
├── CreditosActivos10.py
├── Rentabilidad.py
├── Detallado de Canje.py
├── Kpis's.py
├── ejecucion_completa.py (opcional)
└── README.md
