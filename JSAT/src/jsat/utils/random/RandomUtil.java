/*
 * Copyright (C) 2017 Edward Raff <Raff.Edward@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package jsat.utils.random;

import java.util.Random;

/**
 * This class provides assorted utilities related to random number generation
 * and use.
 *
 * @author Edward Raff <Raff.Edward@gmail.com>
 */
public class RandomUtil
{

    private RandomUtil()
    {
    }
    
    /**
     * If not specified, this will be the seed used for random objects used
     * internally within JSAT.<br>
     * <br>
     * You may change this to any desired value at the start of any experiments
     * to consistently get new experimental results. You may also change it at
     * any point without ill effect, but the purpose is for consistent and
     * repeatable experiments.
     */
    public static int DEFAULT_SEED = 963863937;
    /**
     * This controls whether or not {@link #getRandom() } will cause a change in
     * the {@link #DEFAULT_SEED} each time it is called. This is the default to
     * ensure that multiple calls to getRandom will not return an equivalent
     * object.
     */
    public static boolean INCREMENT_SEEDS = true;

    /**
     * A large prime value to increment by so that we can take many steps before
     * repeating.
     */
    private static final int SEED_INCREMENT = 1506369103;

    /**
     * Returns a new Random object that can be used. The Random object returned
     * is promised to be a reasonably high quality PSRND with low memory
     * overhead that is faster to sample from than the default Java
     * {@link Random}. Essentially, it will be better than Java's default PRNG
     * in every way. <br>
     * <br>
     * By default, multiple calls to this method will result in multiple,
     * different, seeds. Controlling the base seed and its behavior can be done
     * using {@link #DEFAULT_SEED} and {@link #INCREMENT_SEEDS}.
     *
     * @return a new random number generator to use. 
     */
    public static synchronized Random getRandom()
    {
        int seed = DEFAULT_SEED;
        if(INCREMENT_SEEDS)
            DEFAULT_SEED += SEED_INCREMENT;
        return new XORWOW(seed);
    }
    
    /**
     * Returns a new Random object that can be used, initiated with the given
     * seed. The Random object returned is promised to be a reasonably high
     * quality PSRND with low memory overhead that is faster to sample from than
     * the default Java {@link Random}. Essentially, it will be better than
     * Java's default PRNG in every way. <br>
     *
     * @param seed the seed of the PRNG, which determines the sequence generated
     * by the returned object
     * @return a new random number generator to use.
     */
    public static synchronized Random getRandom(int seed)
    {
        return new XORWOW(seed);
    }
}
