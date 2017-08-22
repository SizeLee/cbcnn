
def combineNumCal(all, take):
    if take>all:
        print('Error in combineNumCal: take is large than all')
        exit(1) #todo throw error

    Anum = 1
    TakeAnum = 1
    for i in range(take):
        Anum *= (all - i)
        TakeAnum *= (take - i)

    return int(Anum/TakeAnum)

if __name__ == '__main__':
    print(combineNumCal(6, 4))
    print(combineNumCal(10, 4))