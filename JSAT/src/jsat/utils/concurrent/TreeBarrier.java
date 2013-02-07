
package jsat.utils.concurrent;

import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Tree Barrier is a barrier that requires only log(n) communication
 * for barrier with n threads. To use this barrier, each thread must 
 * know its own unique ID that is sequential in [0, number of threads)
 * <br><br>
 * NOTE: The Tree Barrier implementation is not safe for accessing 
 * multiple times in a row. If accessed while some threads are still
 * attempting to exit, corruption and deadlock can occur. If needed,
 * two TreeBarriers can be used by alternating back and forth. 
 * 
 * @author Edward Raff
 */
public class TreeBarrier 
{
    final private int parties;
    private Lock[] locks;
    private volatile boolean competitionCondition;

    /**
     * Creates a new Tree Barrier for synchronization
     * @param parties the number of threads that must arrive to synchronize
     */
    public TreeBarrier(int parties)
    {
        this.parties = parties;
        locks = new Lock[parties-1];
        for(int i = 0; i < locks.length; i++)
            locks[i] = new ReentrantLock(false);
        competitionCondition = true;
    }
    
    /**
     * Waits for all threads to reach this barrier. 
     * 
     * @param ID the id of the thread attempting to reach the barrier. 
     * @throws InterruptedException if one of the threads was interrupted 
     * while waiting on the barrier
     */
    public void await(int ID) throws InterruptedException
    {
        if(parties == 1)//what are you doing?!
            return;
        final boolean startCondition = competitionCondition;
        int competingFor = (locks.length*2-1-ID)/2;
        
        while (competingFor >= 0)
        {
            final Lock node = locks[competingFor];
            if (node.tryLock())//we lose, must wait
            {
                synchronized (node)//ignore warning, its correct. We are using the lock both for competition AND to do an internal wait
                {
                    while(competitionCondition == startCondition)
                        node.wait();
                }
                node.unlock();
                
                wakeUpTarget(competingFor*2+1);
                wakeUpTarget(competingFor*2+2);
                return;
            }
            else //we win, comete for another round!
            {
                if(competingFor == 0)
                    break;//we have won the last round!
                competingFor = (competingFor-1)/2;
            }
        }
        
        //We won! Inform the losers
        competitionCondition = !competitionCondition;
        wakeUpTarget(0);//biggest loser
    }
    
    private void wakeUpTarget(int nodeID)
    {
        if(nodeID < locks.length)
        {
            synchronized(locks[nodeID])
            {
                locks[nodeID].notify();
            }
        }
    }
}
