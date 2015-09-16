package jsat.text.tokenizer;

import static org.junit.Assert.assertEquals;

import java.util.Arrays;
import java.util.List;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 *
 * @author Edward Raff
 */
public class NGramTokenizerTest {

  @BeforeClass
  public static void setUpClass() {
  }

  @AfterClass
  public static void tearDownClass() {
  }

  public NGramTokenizerTest() {
  }

  @Before
  public void setUp() {
  }

  @After
  public void tearDown() {
  }

  /**
   * Test of tokenize method, of class NGramTokenizer.
   */
  @Test
  public void testTokenize_String() {
    System.out.println("tokenize");
    final String input = "the dog barked";
    final NaiveTokenizer naiveToken = new NaiveTokenizer();
    final NGramTokenizer instance = new NGramTokenizer(3, naiveToken, true);
    final List<String> expResult = Arrays.asList("the", "dog", "barked", "the dog", "dog barked", "the dog barked");
    final List<String> result = instance.tokenize(input);

    assertEquals(expResult.size(), result.size());
    for (int i = 0; i < expResult.size(); i++) {
      assertEquals(expResult.get(i), result.get(i));
    }
  }

}
