package jsat.utils.concurrent;

import static java.lang.Math.min;

/**
 *
 * @author Edward Raff
 */
public class ParallelUtils {

  /**
   * Gets the ending index (exclusive) for splitting up a list of items into
   * {@code P} evenly sized blocks. In the event that {@code N} is not evenly
   * divisible by {@code P}, the size of ranges will differ by at most 1.
   *
   * @param N
   *          the number of items to split up
   * @param ID
   *          the block number in [0, {@code P})
   * @param P
   *          the number of blocks to break up the items into
   * @return the ending index (exclusive) of the blocks owned by the {@code ID}
   *         'th process.
   */
  public static int getEndBlock(final int N, final int ID, final int P) {
    final int rem = N % P;
    final int start = N / P * (ID + 1) + min(rem, ID + 1);
    return start;
  }

  /**
   * Gets the starting index (inclusive) for splitting up a list of items into
   * {@code P} evenly sized blocks. In the event that {@code N} is not evenly
   * divisible by {@code P}, the size of ranges will differ by at most 1.
   *
   * @param N
   *          the number of items to split up
   * @param ID
   *          the block number in [0, {@code P})
   * @param P
   *          the number of blocks to break up the items into
   * @return the starting index (inclusive) of the blocks owned by the
   *         {@code ID}'th process.
   */
  public static int getStartBlock(final int N, final int ID, final int P) {
    final int rem = N % P;
    final int start = N / P * ID + min(rem, ID);
    return start;
  }
}
