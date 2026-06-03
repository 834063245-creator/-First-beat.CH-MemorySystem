"""Add LITE_DISABLE_* to settings.py"""
with open('app/config/settings.py', 'r', encoding='utf-8') as f:
    content = f.read()

old = 'DEPLOY_MODE = os.getenv("DEPLOY_MODE", "full")\nIS_LITE = DEPLOY_MODE == "lite"\nLITE_WORK_MEMORY_BUDGET = 5000 if IS_LITE else 50000'
new = 'DEPLOY_MODE = os.getenv("DEPLOY_MODE", "full")\nIS_LITE = DEPLOY_MODE == "lite"\n\n# 轻量版功能开关\nLITE_DISABLE_BACKGROUND_TASKS = True        # 禁用后台巩固 + 空闲回顾\nLITE_DISABLE_IMPULSE = True                 # 禁用冲动调度器 + 独立开口\nLITE_WORK_MEMORY_BUDGET = 5000 if IS_LITE else 50000'

assert old in content, 'pattern not found'
content = content.replace(old, new)
with open('app/config/settings.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('OK')
