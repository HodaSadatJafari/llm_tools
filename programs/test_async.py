import asyncio


async def do_some_work(num):
    print(f"Starting work {num}")
    await asyncio.sleep(1)
    print(f"Work complete {num}")


async def do_a_lot_of_work_in_parallel():
    await asyncio.gather(do_some_work(1), do_some_work(2), do_some_work(3))


if __name__ == "__main__":
    asyncio.run(do_a_lot_of_work_in_parallel())
