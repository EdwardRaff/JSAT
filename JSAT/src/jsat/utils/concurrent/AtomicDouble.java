
package jsat.utils.concurrent;

import java.util.concurrent.atomic.AtomicLong;
import static java.lang.Double.doubleToRawLongBits;
import static java.lang.Double.longBitsToDouble;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;
import java.util.function.LongUnaryOperator;

/**
 *
 * @author Edward Raff
 */
final public class AtomicDouble 
{
    private final AtomicLong base;

    public AtomicDouble(double value)
    {
        base = new AtomicLong();
        set(value);
    }
    
    public void set(double val)
    {
        base.set(Double.doubleToRawLongBits(val));
    }
    
    public double get()
    {
        return longBitsToDouble(base.get());
    }
    
    public double getAndAdd(double delta)
    {
        while(true)
        {
            double orig = get();
            double newVal = orig + delta;
            if(compareAndSet(orig, newVal))
                return orig;
        }
    }
    
    /**
     * Atomically adds the given value to the current value.
     *
     * @param delta the value to add
     * @return the updated value
     */
    public final double addAndGet(double delta) 
    {
        while(true) 
        {
            double orig = get();
            double newVal = orig + delta;
            if (compareAndSet(orig, newVal))
                return newVal;
        }
    }

    /**
     * Atomically updates the current value with the results of applying the
     * given function, returning the updated value. The function should be
     * side-effect-free, since it may be re-applied when attempted updates fail
     * due to contention among threads.
     *
     * @param updateFunction a side-effect-free function
     * @return the updated value
     */
    public final double updateAndGet(DoubleUnaryOperator updateFunction)
    {
        double prev, next;
        do
        {
            prev = get();
            next = updateFunction.applyAsDouble(prev);
        }
        while (!compareAndSet(prev, next));
        return next;
    }

    /**
     * Atomically updates the current value with the results of applying the
     * given function, returning the previous value. The function should be
     * side-effect-free, since it may be re-applied when attempted updates fail
     * due to contention among threads.
     *
     * @param updateFunction a side-effect-free function
     * @return the previous value
     */
    public final double getAndUpdate(DoubleUnaryOperator updateFunction)
    {
        double prev, next;
        do
        {
            prev = get();
            next = updateFunction.applyAsDouble(prev);
        }
        while (!compareAndSet(prev, next));
        return prev;
    }
    
    /**
     * Atomically updates the current value with the results of
     * applying the given function to the current and given values,
     * returning the previous value. The function should be
     * side-effect-free, since it may be re-applied when attempted
     * updates fail due to contention among threads.  The function
     * is applied with the current value as its first argument,
     * and the given update as the second argument.
     *
     * @param x the update value
     * @param accumulatorFunction a side-effect-free function of two arguments
     * @return the previous value
     */
    public final double getAndAccumulate(double x, DoubleBinaryOperator accumulatorFunction)
    {
        double prev, next;
        do
        {
            prev = get();
            next = accumulatorFunction.applyAsDouble(prev, x);
        }
        while (!compareAndSet(prev, next));
        return prev;
    }
    
    /**
     * Atomically updates the current value with the results of
     * applying the given function to the current and given values,
     * returning the updated value. The function should be
     * side-effect-free, since it may be re-applied when attempted
     * updates fail due to contention among threads.  The function
     * is applied with the current value as its first argument,
     * and the given update as the second argument.
     *
     * @param x the update value
     * @param accumulatorFunction a side-effect-free function of two arguments
     * @return the updated value
     */
    public final double accumulateAndGet(double x, DoubleBinaryOperator accumulatorFunction)
    {
        double prev, next;
        do
        {
            prev = get();
            next = accumulatorFunction.applyAsDouble(prev, x);
        }
        while (!compareAndSet(prev, next));
        return next;
    }
    
    public boolean compareAndSet(double expect, double update)
    {
        return base.compareAndSet(doubleToRawLongBits(expect), doubleToRawLongBits(update));
    }
    
    public boolean weakCompareAndSet(double expect, double update)
    {
        return base.weakCompareAndSet(doubleToRawLongBits(expect), doubleToRawLongBits(update));
    }

    @Override
    public String toString()
    {
        return ""+get();
    }
    
    
}
