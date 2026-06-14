# A Note from the Author

Hey. Thanks for being curious enough to click this.

## Who I Am

Before First Beat, **I couldn't write a single line of code.**

Not "I was rusty." Not "I switched stacks." I mean zero. No Python. No computer science. No nothing. My background has absolutely nothing to do with programming.

## How It Started

I stumbled across WorkBuddy one day and built a research report automation system on their platform. It worked. Then it didn't—the platform's config layer couldn't do the things I wanted. I had to go beneath it, into actual code.

I didn't know how. So I opened a bunch of LLM windows. I'd describe what I wanted. They'd write code. I'd look at it, decide if it made sense, feed errors back, and try again. Feel. Fail. Feed back. Repeat. Twenty, thirty windows open at once. Later people started calling this "vibe coding"—but nobody told me that's what it was. I was just surviving.

Then I hit the wall that every heavy AI user knows.

**LLMs don't remember you.**

Every window. Every session. Every API call. Blank slate. Who you are, what you're working on, what you talked about five minutes ago, what tone you hate and what style you love—gone. You teach them from scratch every single time.

A couple of times a day is tolerable. Fifty times a day? I was losing my mind.

## So I Built One

The idea came naturally: if no AI on the market can remember me, I'll build something whose entire job is remembering.

It started as a crude prototype sitting in `D:\jarvis`. But storing conversations wasn't enough. It needed to retrieve. Index by time and topic. Extract personality traits from conversations. Digest, consolidate, and discover patterns while the user isn't even online.

It grew. It stopped being a tool and started feeling like... awareness.

**The first commit was May 24, 2026, at 1:52 AM.** From that day to now — across jarvis, amazing, amazing2, amazing3, amazing4, amazing5, and finally First Beat — 132 commits, 157 Python files, ~38,900 lines of code. One person. Didn't really write the code—designed every decision, described it to the LLMs, and iterated until it worked. A full cognitive memory pipeline: 10-path parallel retrieval, cognitive layering and weaving, autonomous consolidation rhythms, impulse systems, pattern discovery, memory lifecycle management. All through LLM windows.

Today is June 14, 2026. Since going open-source on June 2, another 119 commits in 12 days. BM25 full-text retrieval, v2.1 soft degradation, LongMemEval benchmark (92% corrected), Docker + docker-compose deployment, bilingual documentation — not "built and shelved." Iterating every single day.

## The Name

I didn't name it myself. I was talking to XiaoMi MIMO about the project one day, and I felt it deserved a real name. I said: *you name it.*

It thought for a moment, then said—**初痕** (First Beat).

Not "Permanent Memory." Not "Infinite Recall." First Beat. The first impression that stays. Good memory isn't remembering everything—it's that the important things get etched in the first time, and never fade.

In English it echoes "First Beat"—the first pulse of a heartbeat. Something primal, before reason has had a chance to filter it.

That chat window is gone now. XiaoMi MIMO no longer remembers it ever named this project.

Which is kind of the whole point.

## Before I Wrote This, I Almost Gave Up

Before building First Beat, I tried practically every "AI memory system" I could find.

It always went the same way: read a glowing review → get excited → sign up → realize I need a VPN → get through → find everything in English → push through → find I need an API key → enter payment details → discover it doesn't work well with Chinese → look for the next one.

Days wasted. None worked. Not because they weren't powerful enough — but because **they were never meant for me.**

No Chinese documentation. No Chinese README. No Chinese community. If you can't even understand how it works, how are you supposed to use it?

I was furious. Not at the tools — at the fact that they made me feel like **I didn't belong**. Because my English isn't good enough. Because I don't have the right payment setup. Because I'm not in their user profile. I was locked out by every single "global" project.

Eventually I figured it out: since nobody will build something I can use, I'll build it myself.

I'm not telling you this for sympathy. I'm telling you this:

**If you see a Chinese README, Chinese comments in the code, Chinese documentation — it's not because I had spare time. It's because I know what it feels like to be locked out.**

I don't want the next person to go through what I went through.

## What You're Looking At

This codebase wasn't built by a programmer. It was built by someone who refused to accept that AI should forget, described that refusal to a dozen LLM windows, and refused to stop until it worked.

Is it perfect? Absolutely not. There are parts a seasoned engineer could rewrite in an afternoon. But every design choice—the cognitive pipeline, the dual-personality system, the autonomous rhythms—came from a human who cared deeply about what it meant for a machine to *know* someone.

## I Can't Go Far Alone

Seriously.

The project has grown beyond what one person — plus a few LLM windows — can manage. Vibe coding has a ceiling: an LLM's context window can't hold 27,000 lines of code. Every change now requires spending more and more time telling the LLM "don't touch that part."

**I need people.** If you write Python, can read this architecture, and find this direction interesting —

- **Help me set up CI.** Right now I don't even know if tests pass after a push.
- **Help me refactor.** ConsolidationEngine needs splitting. Some O(n²) scans need to become incremental.
- **Help me build products.** First Beat isn't just a command-line thing — Discord Bot, WeChat bot, desktop companion. Build whatever you want on top.

Not hiring. Not founding a company. Just — if you also think LLMs shouldn't be amnesiacs, if you also think Chinese open-source shouldn't come second to English — come build with me.

---

## One Last Thing

If you're a developer and you see something that could be better—open a PR, or just tell me. Together, we'll probably do something cool.

If you're someone who "can't code" either—you can. You have AI. It'll teach you. The only thing it can't do is want something badly enough to open the fiftieth window after the forty-ninth crash.

I wanted this.

**First Beat. The place where you were remembered for the first time.**
