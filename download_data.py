from __future__ import print_function

import os

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

#URLBASE = 'https://storage.ramp.studio/storm_forecast/{}'
#DATA = [
#    'train.csv', 'test.csv']


def main(output_dir='data'):
    #filenames = DATA
    #urls = [URLBASE.format(filename) for filename in filenames]

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print('\n!READ ME! \nDue to maintenance reason, please download manually '
          'the test.csv and train.csv files using this link:')
    print('\nhttps://filesender.renater.fr/?s=download&token=5318b1e5-0fe9-4ca2-4f83-054dd049300c#')
    print('\nTHEN: store them in the storm_forecast/data/ folder that has just been created.')
    print('\nThanks!')
    # notfound = []
    #for url, filename in zip(urls, filenames):
    #    output_file = os.path.join(output_dir, filename)

    #    if os.path.exists(output_file):
     #       continue

      #  print("Downloading from {} ...".format(url))
      #  urlretrieve(url, filename=output_file)
      #  print("=> File saved as {}".format(output_file))



if __name__ == '__main__':
    test = os.getenv('RAMP_TEST_MODE', 0)

    if test:
        print("Testing mode, not downloading any data.")
    else:
        main()
