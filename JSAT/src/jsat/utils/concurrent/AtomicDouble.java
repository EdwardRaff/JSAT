
package jsat.utils.concurrent;

import java.util.concurrent.atomic.AtomicLong;
import static java.lang.Double.doubleToRawLongBits;
import static java.lang.Double.longBitsToDouble;

/**
 *
 * @author Edward Raff
 */
final public class AtomicDouble 
{
    private final AtomicLong base;

    public AtomicDouble(final double value)
    {
        base = new AtomicLong();
        set(value);
    }
    
    public void set(final double val)
    {
        base.set(Double.doubleToRawLongBits(val));
    }
    
    public double get()
    {
        return longBitsToDouble(base.get());
    }
    
    public double getAndAdd(final double delta)
    {
        while(true)
        {
            final double orig = get();
            final double newVal = orig + delta;
            if(compareAndSet(orig, newVal)) {
              return orig;
            }
        }
    }
    
    /**
     * Atomically adds the given value to the current value.
     *
     * @param delta the value to add
     * @return the updated value
     */
    public final double addAndGet(final double delta) 
    {
        while(true) 
        {
            final double orig = get();
            final double newVal = orig + delta;
            if (compareAndSet(orig, newVal)) {
              return newVal;
            }
        }
    }
    
    public boolean compareAndSet(final double expect, final double update)
    {
        return base.compareAndSet(doubleToRawLongBits(expect), doubleToRawLongBits(update));
    }
    
    public boolean weakCompareAndSet(final double expect, final double update)
    {
        return base.weakCompareAndSet(doubleToRawLongBits(expect), doubleToRawLongBits(update));
    }

    @Override
    public String toString()
    {
        return ""+get();
    }
    
    
}
