# Git 入门文档

面向 AI 辅助 Git 仓库管理的测试学习项目，梳理常用 Git 命令与核心概念。

---

## 0. 三个工作区（第 6 点）

Git 把文件分成三个区域，理解它们是理解所有命令的基础：

```
工作区 (Working Directory)      你正在编辑的文件，磁盘上真实的样子
        │  git add
        ▼
暂存区 (Staging Area / Index)   准备下一次提交的快照
        │  git commit
        ▼
本地仓库 (Local Repository)      .git 里的提交历史
        │  git push
        ▼
远程仓库 (Remote Repository)     GitHub / GitLab 等服务器上的副本
```

| 区域 | 说明 | 常见命令 |
|---|---|---|
| 工作区 Working Directory | 当前可见、可编辑的文件 | `git status` 查看改动 |
| 暂存区 Staging Area | `git add` 后进入，等待提交 | `git add <file>` |
| 本地仓库 Local Repository | `git commit` 后形成提交，HEAD 指向它 | `git log` |
| 远程仓库 Remote Repository | 团队共享，`push`/`pull` 同步 | `git push` / `git fetch` |

反向操作（撤销方向）：

```
工作区 ←── git restore <file> ── 暂存区 ←── git restore --staged ── 本地仓库
```

---

## 1. discard / reset / revert 的区别

三者都涉及"撤销"，但作用对象和原理不同：

| | 作用对象 | 是否改历史 | 可恢复性 | 适用场景 |
|---|---|---|---|---|
| **discard** | 未提交的改动（工作区/暂存区） | 否 | 不可恢复 | 放弃本地改动 |
| **reset** | 提交 / 分支指针（HEAD） | 是 | reflog 可救 | 本地整理历史 |
| **revert** | 已提交的改动 | 否（新增反向提交） | 安全 | 撤销已推送的提交 |

### discard（丢弃）

- 不是独立 Git 命令，是 GUI（VS Code、GitHub Desktop）里的功能。
- 底层多为 `git checkout -- <file>` 或 `git restore <file>`。
- 直接让文件回到上次提交状态，**改动从未进入版本库，不可恢复**。

```bash
git restore file.txt            # 丢弃工作区改动
git restore --staged file.txt   # 从暂存区撤回（保留工作区改动）
```

### reset（重置）

移动分支指针 HEAD 到目标提交，**会重写历史**。

```bash
git reset --soft  <commit>   # 只移动 HEAD，改动留在暂存区
git reset --mixed <commit>   # 默认；改动退回工作区
git reset --hard  <commit>   # 连工作区一起清掉（危险！）
```

> 已 push 的提交若被 reset，需要 force push，会影响协作者。
> 误操作后可用 `git reflog` 找回旧提交。

### revert（还原）

针对已提交内容，**新建一个"反向提交"抵消目标提交的改动**，不改历史。

```bash
git revert <commit>     # 生成一个新提交，抵消该提交的改动
```

> 记忆口诀：**未提交用 discard，本地重写用 reset，公共历史用 revert**。

---

## 2. diff（查看差异）

`git diff` 比较不同区域/版本间的差异。

```bash
git diff                      # 工作区 vs 暂存区
git diff --staged             # 暂存区 vs 本地最新提交（= git diff --cached）
git diff HEAD                 # 工作区 vs 最新提交
git diff <commitA> <commitB>  # 两个提交之间
git diff <branchA>..<branchB> # 两个分支之间
git diff <commit> -- <path>   # 只看某文件/目录
```

常用参数：

```bash
git diff --stat      # 只看统计（增删行数），不看具体内容
git diff <A> <B>     # 注意：A 是"旧"，B 是"新"
```

`git log <A>..<B>` 则列出 B 有而 A 没有的提交（配合看历史）。

---

## 3. merge（合并）

把一个分支的改动并入当前分支。

```bash
git switch main          # 先切到目标分支
git merge feature        # 把 feature 合入 main
```

### 两种合并结果

**Fast-forward（快进）**：目标分支没有新提交，直接移动指针，不产生合并提交。

```
合并前:  main→A        feature→A←B←C
合并后:  main→A←B←C
```

**Three-way merge（三方合并）**：两边都有新提交，产生一个**合并提交**。

```
        main: A←D
                 ↘
合并提交 M ←──────── （合并 D 和 C）
                 ↗
     feature: A←C
```

```bash
git merge --no-ff feature   # 强制生成合并提交（保留分支历史形状）
git merge --abort           # 合并冲突时放弃合并
```

### 删除已合并的分支

```bash
git branch -d feature       # 安全删除（仅当已合并）
git branch -D feature       # 强制删除（未合并也删）
git push origin --delete feature   # 删除远程分支
```

---

## 4. delete 分支

```bash
git branch -d <branch>          # 删除本地分支（已合并才允许）
git branch -D <branch>          # 强制删除本地分支
git push origin --delete <branch>  # 删除远程分支
git push origin :<branch>       # 等价写法
```

> 删除分支只是删掉一个"指针"，提交内容只要还被其他分支/标签引用就不会丢。

---

## 5. conflict（冲突）

当两个分支修改了**同一文件的同一位置**，Git 无法自动决定保留哪个，产生冲突。

冲突时文件里会出现标记：

```
<<<<<<< HEAD
当前分支的内容
=======
传入分支的内容
>>>>>>> feature
```

解决流程：

```bash
git merge feature        # 提示 CONFLICT
# 1. 手动编辑文件，保留想要的内容，删除 <<<< ==== >>>> 标记
git add <file>           # 2. 标记为已解决
git merge --continue     # 3. 完成合并（或 git commit）
# 或者放弃：
git merge --abort
```

```bash
git status               # 查看哪些文件冲突（both modified）
git diff                 # 查看冲突细节
```

> rebase 冲突用 `git rebase --continue` / `git rebase --abort`。

---

## 6. working directory / local / remote repository

见 **第 0 节**。核心区分：

- **working directory**：你改的东西。
- **local repository**：你本地的 .git 历史。
- **remote repository**：远端共享历史。

常用同步命令：

```bash
git remote -v            # 查看远程地址
git fetch origin         # 拉取远程更新（不改工作区）
git pull origin main     # fetch + merge
git push origin main     # 推送本地提交
```

---

## 7. cherry-pick（挑选提交）

把**某个提交**单独复制到当前分支（只挑一个/几个，不合并整条分支）。

```bash
git cherry-pick <commit>              # 复制单个提交
git cherry-pick <c1> <c2> <c3>        # 复制多个
git cherry-pick <start>^..<end>       # 复制一个范围
git cherry-pick --continue / --abort  # 冲突处理
```

用途：只想拿别的分支上的某个修复，而不合并它全部改动。

> 会生成**新的提交哈希**（内容相同但不是同一个 commit）。

---

## 8. stash（暂存现场）

临时保存未提交的改动，把工作区清干净，腾出手做别的事。

```bash
git stash                 # 保存当前改动并回到干净状态
git stash -u              # 连同未跟踪文件一起保存
git stash list            # 查看 stash 列表
git stash apply           # 恢复最新 stash（保留 stash 记录）
git stash pop             # 恢复并删除该 stash 记录
git stash drop            # 删除某条 stash
git stash show -p         # 查看 stash 内容 diff
```

**典型场景**：正改着代码，需要紧急切分支修 bug。

```bash
git stash
git switch hotfix
# ... 修复并提交 ...
git switch main
git stash pop
```

---

## 9. rebase（变基）与 merge 的区别

把当前分支的提交"搬"到目标分支的最新提交之后，**形成一条直线历史**。

```bash
git switch feature
git rebase main
```

过程：先找到两分支共同祖先，把 feature 上的提交逐个在 main 最新提交上重放。

```
merge 结果（有分叉 + 合并提交）:
        A←D←M       main
           ↖ ↗
            C       feature

rebase 结果（一条直线）:
        A←D←C'      （C' 是重放后的新提交）
```

### rebase vs merge

| | merge | rebase |
|---|---|---|
| 历史形状 | 有分叉，保留合并点 | 一条直线，干净 |
| 是否产生新提交 | 产生合并提交 | 重写提交（新哈希） |
| 是否改历史 | 不改 | **改历史** |
| 安全性 | 已推送分支可安全 merge | 已推送分支 rebase 需 force push，危险 |
| 冲突处理 | 一次解决 | 每个提交可能都要解决 |

**使用原则**：

- 公共分支（已推送、多人共享）→ 用 **merge**。
- 个人本地分支整理 → 用 **rebase**。
- 黄金法则：**不要 rebase 已经推送到远程、别人可能基于它工作的提交**。

```bash
git pull --rebase        # 拉取时用 rebase 方式，避免多余的合并提交
```

---

## 附录：本次实操速查

```bash
git reset --hard <commit>       # 回退到某提交（丢弃后续提交）
git reflog                      # 找回被 reset 掉的提交
git switch <branch>             # 切换分支
git merge <branch>              # 合并分支
git branch -D <branch>          # 强制删除本地分支
git push origin --delete <branch>  # 删除远程分支
git diff <A> <B> --stat         # 比较两个提交
```
