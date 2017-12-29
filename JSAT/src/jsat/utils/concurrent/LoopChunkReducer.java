/*
 * Copyright (C) 2017 Edward Raff
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
package jsat.utils.concurrent;

import java.util.function.BinaryOperator;

/**
 * This functional interface is similar to that of {@link LoopChunkReducer},
 * allowing convieniently processing of a range of values for parallel
 * computing. However, the Reducer returns an object implementing the
 * {@link BinaryOperator} interface. The goal is that all chunks will eventually
 * be merged into a single result. This interface is preffered over using normal
 * streams to reduce unecessary object creation and reductions.
 *
 * @author Edward Raff
 * @param <T> The object type that the Loop Chunk Reducer will return
 */
public interface LoopChunkReducer<T>
{
    /**
     * Runs a process over the given loop portion, returning a single object of type {@link T}. 
     * @param start the starting index to process 
     * @param end the ending index to process
     * @return the object to return 
     */
    public T run(int start, int end);
}
