import os
import pty
import subprocess
import pandas as pd
import numpy as np
import argparse

argparser = argparse.ArgumentParser()
# this only produces a score on the training set of COGS by default
argparser.add_argument("--num_train_examples_to_check", type=int, default=5) # I like people running demos to see the score output quick and then they can do a longer run later if they want
argparser.add_argument("--use_dev_split", action="store_true")
argparser.add_argument("--use_gen_split", action="store_true")
argparser.add_argument("--use_test_split", action="store_true")
argparser.add_argument("--cp_examples_only", action="store_true")
# if Google Colab or other environment crashes, you may want to pick up where it left off
argparser.add_argument("--skip_rows", type=int, default=0)
argparser.add_argument("--do_pp_recursion_gen_split", action="store_true") # needs to be done separately as very slow
argparser.add_argument("--do_cp_recursion_gen_split", action="store_true") # needs to be done separately as very slow
args = argparser.parse_args()
if (args.use_gen_split and args.use_dev_split) or (args.use_gen_split and args.use_test_split) or (args.use_dev_split and args.use_test_split):
  print("Please select just one of the arguments `--use_gen_split`,`--use_dev_split`, or `--use_test_split`")
  raise Exception("Please select just one of the arguments `--use_gen_split`,`--use_dev_split`,`--use_test_split`")

if args.do_pp_recursion_gen_split and not args.use_gen_split:
  print("`do_pp_recursion_gen_split` can only be used when `--use_gen_split` is used!")
  raise Exception("`do_pp_recursion_gen_split` can only be used when `--use_gen_split` is used!")

if args.do_cp_recursion_gen_split and not args.use_gen_split:
  print("`do_cp_recursion_gen_split` can only be used when `--use_gen_split` is used!")
  raise Exception("`do_cp_recursion_gen_split` can only be used when `--use_gen_split` is used!")

if args.do_cp_recursion_gen_split and args.do_pp_recursion_gen_split:
  print("`do_cp_recursion_gen_split` cannot be used with `do_pp_recursion_gen_split` at this time, due to slowness of these recursion splits we do just one at a time separate")
  raise Exception("`do_cp_recursion_gen_split` cannot be used with `do_pp_recursion_gen_split` at this time, due to slowness of these recursion splits we do just one at a time separate")

if args.cp_examples_only and (args.do_cp_recursion_gen_split or args.do_pp_recursion_gen_split):
  print("`--cp_examples_only` not supported in combination with gen split selections, e.g. `--do_pp_recursion_gen_split` or `--do_cp_recursion_gen_split`.")
  raise Exception("`--cp_examples_only` not supported in combination with gen split selections, e.g. `--do_pp_recursion_gen_split` or `--do_cp_recursion_gen_split`.")


base_path = os.path.abspath(".")
# Load dependency if not available.
# we use the Restricted Access Sequence Processing interpreter from "Thinking Like Transformers" Weiss et al 2021 ( https://arxiv.org/abs/2106.06981 )
# This RASP is an academic language which can be theoretically compiled to Transformer neural network weights, useful for thinking about Transformer capabilities.
print("Note the RASP dependency requires python3.10-venv and its own dependencies; if there are errors please check the RASP dependencies' instructions (and run `apt install python3.10-venv` or equivalent for your operating system)")
if not os.path.exists(base_path + "/RASP"):
  subprocess.run("git clone https://github.com/tech-srl/RASP.git", shell=True, executable='/bin/bash')
  os.chdir(base_path + "/RASP")
  subprocess.run(base_path + "/RASP/setup.sh", shell=True, executable="/bin/bash")
  os.chdir(base_path)

with open(base_path + "/RASP/rasp2.sh", "w") as f:
  f.write("""#!/bin/bash
source raspenv/bin/activate
python3 -m RASP_support
deactivate
""")
os.chmod(base_path + "/RASP/rasp2.sh", 750)

os.chdir(base_path + "/RASP")
main, secondary = pty.openpty()
proc = subprocess.Popen(base_path + "/RASP/rasp2.sh", shell=True, executable='/bin/bash', stdin=subprocess.PIPE, stdout=secondary)
print(proc)
stdin_handle = proc.stdin
os.set_blocking(main, False)
stdout_handle = os.fdopen(main)
os.chdir(base_path)
while len(stdout_handle.readlines()) == 0:
  pass # on some dependency versions
# there is some delay before RASP process responds, this avoids flooding it with RASP input 
# before it is ready
with open(base_path + "/" + "rasp-for-cogs.rasp", "r") as f:
  rasp_setup_lines = f.readlines()
rasp_setup_lines.append("\n")
input_lines = [bytes(line, 'utf8') for line in rasp_setup_lines]

# the RASP REPL is meant to be used in an interactive mode, and we are feeding it large batches of input which is outside its design
# this leads to some flakiness loading the code into the RASP interpreter
# so let us load it 5 lines at a time and flush out the output at each step
batch_idx = 0
while (batch_idx+1)*5 < len(input_lines) + 5:
  lines_to_write = input_lines[batch_idx*5:min((batch_idx+1)*5, len(input_lines))]
  #print(lines_to_write)
  stdin_handle.writelines(lines_to_write)
  stdin_handle.flush()
  stdout_handle.flush()
  outputlines = stdout_handle.readlines()
  #print(f"output: {outputlines}")
  batch_idx += 1

running_scores_logfile_handle = open("cogs_examples_in_rasp_running_scores.log", "w")

# discard all output from running the setup part of the script
input_lines = ["output;\n", "autoregressive_output = output;\n"]
input_lines = [bytes(line, 'utf8') for line in input_lines]
stdin_handle.writelines(input_lines)
stdin_handle.flush()
outputline = stdout_handle.readline()
while outputline.find("Example: autoregressive_output") < 0:
  outputline = stdout_handle.readline()

def process_example(example, suppress_output=True, debug_mode=False):
  current_example = example.split(" ")
  if "|" not in current_example:
      current_example.append("|")
  nextcharacter = " "
  while nextcharacter != "":
    next_input = f"set example {current_example}\nautoregressive_output;\n".replace("'", '"')
    input_lines = [next_input]
    input_lines = [bytes(line, 'utf8') for line in input_lines]
    stdin_handle.writelines(input_lines)
    stdin_handle.flush()
    outputline = stdout_handle.readline()
    while outputline.find("Example: autoregressive_output") < 0:
      if debug_mode:
        print(f"skipping: {outputline}")
      outputline = stdout_handle.readline()
    if not suppress_output:
      print(outputline)
    nextcharacter = outputline.split("=")[1].split("m[")[1].split("]")[0]
    current_example.append(nextcharacter)
  translation = " ".join(current_example).split("|")
  print(f"{translation[0]}\n{translation[1]}")
  return translation[1]

def print_stdout_and_file(string_to_print, file_handle):
  print(string_to_print)
  print(string_to_print, file=file_handle)
  file_handle.flush()


print("""

Note, it is simpler and more performant to just train the Transformer on examples!
This is an academic exercise, writing a neural network compatible program
by hand in the Restricted Access Sequence Processing (compilable to Transformer)
language (Weiss et al 2021, https://arxiv.org/abs/2106.06981 ) to attempt to
prove a Transformer can perform a particular type of solution.

""")

print("Run one example (two semantically identical but syntactically different forms) before loading the dataset:")

process_example("a boy painted the girl", False)

process_example("the girl was painted by a boy", False)
cogs_datafile = None

score_on_train_sample = not args.use_dev_split and not args.use_gen_split and not args.use_test_split

def get_clopper_pearson_confidence_interval(n, k):
  alpha = 0.05
  from scipy.stats import beta
  # Reference: https://en.wikipedia.org/w/index.php?title=Binomial_proportion_confidence_interval&oldid=1252517214#Clopper%E2%80%93Pearson_interval
  # Wikipedia's underlying reference for the beta distribution form https://arxiv.org/abs/1303.1288 equation 4 is also useful,
  cp_confidence_interval = beta.ppf([alpha/2.0, 1-alpha/2.0], [k, k+1],[n-k + 1, n-k])
  # Below https://arxiv.org/abs/1303.1288 eqn 4 they discuss the n == k and k == 0 cases, 
  # which justify the following assignments below and the use of alpha/2.0 (two-tailed test adjustment) above even when we find that k==n or k==0.
  # they give a closed form for these special cases but one can check it is what beta.ppf (which covers all cases) will return there as well.
  if n == k:
    cp_confidence_interval[1] = 1.0
  if k == 0:
    cp_confidence_interval[0] = 0.0
  return cp_confidence_interval

def get_percentages_with_ci_groupby_binary_data(df, groupby_key, alpha=0.05, print_result=False, file_handle=None):
  dfgb = df.groupby(groupby_key)
  c = dfgb.count()
  s = dfgb.sum()
  c.columns = ["count"]
  ci_lows = {}
  ci_highs = {}
  p = {}
  for idx in c.index:
    n = c.loc[idx].values[0]
    k = s.loc[idx].values[0]
    from scipy.stats import beta
    ci = get_clopper_pearson_confidence_interval(n, k)
    ci_lows[idx] = ci[0]*100.0
    ci_highs[idx] = ci[1]*100.0
    p[idx] = float(k)/float(n)*100
    if print_result:
      print_stdout_and_file(f"{idx}: {p[idx]:0.2f}% ({(1-alpha)*100:0.2f}% confidence interval: {ci_lows[idx]:0.2f}% to {ci_highs[idx]:0.2f}% ({k} out of {n}))", file_handle)

  c.insert(0, "hits", s)
  c.insert(0, "percentage", p)
  c.insert(0, "percentage_ci_high", ci_highs)
  c.insert(0, "percentage_ci_low", ci_lows)
  return c

optional_cp_filter = "grep 'that' |" if args.cp_examples_only else ""
filename = None
if score_on_train_sample:
  print("Now load official Kim Linzen 2020 COGS training examples\n(sample from https://raw.githubusercontent.com/najoungkim/COGS/86c9be5929c3b86b510afdfe56a4df9d5d341ab9/data/train.tsv , associated with https://aclanthology.org/2020.emnlp-main.731 )")
  filename = "train_in_distribution_no_augmentations.tsv"
  subprocess.run("wget https://raw.githubusercontent.com/najoungkim/COGS/86c9be5929c3b86b510afdfe56a4df9d5d341ab9/data/train.tsv", shell=True)
  subprocess.run(f"echo 'COGS Sentence	COGS Logical Form	Distribution' > {filename}", shell=True)
  subprocess.run(f"cat train.tsv | grep 'in_distribution' | {optional_cp_filter} grep -v 'sprinkle' >> {filename}", shell=True)
else:
  if args.use_dev_split:
    print("Using dev split, the `num_train_examples_to_check` argument will be ignored")
    print("Now load official Kim Linzen 2020 COGS dev split\n(https://raw.githubusercontent.com/najoungkim/COGS/86c9be5929c3b86b510afdfe56a4df9d5d341ab9/data/dev.tsv , associated with https://aclanthology.org/2020.emnlp-main.731 )")
    filename = "dev_with_header.tsv"
    # one of official author's datasets for COGS paper
    subprocess.run("wget https://raw.githubusercontent.com/najoungkim/COGS/86c9be5929c3b86b510afdfe56a4df9d5d341ab9/data/dev.tsv", shell=True)
    subprocess.run(f"echo 'COGS Sentence	COGS Logical Form	Distribution' > {filename}", shell=True)
    if len(optional_cp_filter) > 0:
      optional_cp_filter = f"| {optional_cp_filter}"
    subprocess.run(f"cat dev.tsv {optional_cp_filter} >> {filename}", shell=True)
  elif args.use_test_split:
    print("Using test split, the `num_train_examples_to_check` argument will be ignored")
    print("Now load official Kim Linzen 2020 COGS test split\n(https://raw.githubusercontent.com/najoungkim/COGS/86c9be5929c3b86b510afdfe56a4df9d5d341ab9/data/test.tsv , associated with https://aclanthology.org/2020.emnlp-main.731)")
    filename = "test_with_header.tsv"
    if len(optional_cp_filter) > 0:
      optional_cp_filter = f"| {optional_cp_filter}"
    # one of official author's dataset for COGS paper
    raise Exception("should not test yet as in development")
    subprocess.run("wget https://raw.githubusercontent.com/najoungkim/COGS/86c9be5929c3b86b510afdfe56a4df9d5d341ab9/data/test.tsv", shell=True)
    subprocess.run(f"echo 'COGS Sentence	COGS Logical Form	Distribution' > {filename}", shell=True)
    subprocess.run(f"cat test.tsv {optional_cp_filter} >> {filename}", shell=True)
  elif args.use_gen_split:
    print("Using gen split, the `num_train_examples_to_check` argument will be ignored")
    print("Now load official Kim Linzen 2020 COGS gen split\n(https://raw.githubusercontent.com/najoungkim/COGS/86c9be5929c3b86b510afdfe56a4df9d5d341ab9/data/gen.tsv , associated with https://aclanthology.org/2020.emnlp-main.731 )")
    raise Exception("should not check gen yet as in development")
    filename = "gen_only_pp_recursion.tsv" if args.do_pp_recursion_gen_split else ("gen_only_cp_recursion.tsv" if args.do_cp_recursion_gen_split else "gen_no_pp_or_cp_recursion.tsv")
    # one of official author's dataset for COGS paper
    subprocess.run("wget https://raw.githubusercontent.com/najoungkim/COGS/86c9be5929c3b86b510afdfe56a4df9d5d341ab9/data/gen.tsv", shell=True)
    subprocess.run(f"echo 'COGS Sentence	COGS Logical Form	Distribution' > {filename}", shell=True)
    if not args.do_pp_recursion_gen_split and not args.do_cp_recursion_gen_split:
      print("Note pp recursion and cp recursion splits (which are slow) are left out by default, run `--do_pp_recursion_gen_split` or `--do_cp_recursion_gen_split` to score one of those at a time separately (they are supported)")
      subprocess.run(f"cat gen.tsv | {optional_cp_filter} grep -v 'pp_recursion' | grep -v 'cp_recursion' >> {filename}", shell=True)
    elif args.do_pp_recursion_gen_split:
      print("Just assessing pp recursion split (which is slow)")
      subprocess.run(f"cat gen.tsv | grep 'pp_recursion' >> {filename}", shell=True)
    elif args.do_cp_recursion_gen_split:
      print("Just assessing cp recursion split (which is slow)")
      subprocess.run(f"cat gen.tsv | grep 'cp_recursion' >> {filename}", shell=True)

print(f"Using prepared datafile: '{filename}' (the filename should describe the dataset you expect to be evaluating with)")
cogs_datafile = pd.read_csv(filename, delimiter="	")

pos_desc = "first"
if args.skip_rows > 0:
  pos_desc = f"first (after {args.skip_rows})"
  print(f"Skipped {args.skip_rows} per argument")
  cogs_datafile = pd.DataFrame(cogs_datafile.values[args.skip_rows:], columns=cogs_datafile.columns)

if score_on_train_sample:
  sentences = cogs_datafile["COGS Sentence"][:args.num_train_examples_to_check]
  lfs_true = cogs_datafile["COGS Logical Form"][:args.num_train_examples_to_check]
  labels = cogs_datafile["Distribution"][:args.num_train_examples_to_check]
else:
  sentences = cogs_datafile["COGS Sentence"]
  lfs_true = cogs_datafile["COGS Logical Form"]
  labels = cogs_datafile["Distribution"]

label = "test" if args.use_test_split else ("dev" if args.use_dev_split else ("gen" if args.use_gen_split else "data"))
disclaimer = "(omitting training data augmentations like sprinkles or preposing as not supported in the grammar of this Restricted Access Sequence Processing Transformer equivalent program and irrelevant to dev,test,gen sets)" if score_on_train_sample else ""

# for replacing out of vocabulary (OOV) words (not present in original COGS train.tsv so cannot be handled as is by model which is not focused on lexical, only structural generalization)
main_train_vocab = set(['a','A','Abigail','Addison','admire','admired','adore','adored','agent','Aiden','Alexander','Amelia','Andrew','Anthony','appreciate','appreciated','Aria','Asher','ate','attempt','attempted','Aubrey','Audrey','Aurora','Ava','Avery','award','awarded','baby','backpack','backyard','bag','bake','baked','bakery','ball','balloon','banana','baron','barrel','basin','basket','bat','bathtub','beach','bean','bear','beast','bed','bee','beer','believe','believed','bell','Bella','bench','Benjamin','beside','bible','bicycle','bike','bin','bird','biscuit','blanket','blender','bless','blessed','block','board','boat','book','booklet','bookstore','bottle','boulder','bowl','box','boy','brain','branch','break','brick','bring','broke','broken','broker','Brooklyn','brought','brush','bucket','bun','bunker','bunny','burn','burned','bush','butterfly','button','buyer','by','cabinet','cafe','cage','cake','Caleb','call','called','Camila','can','canvas','captain','car','carpet','cart','Carter','casket','cassette','castle','cat','ccomp','chair','chalk','champion','change','changed','Charlie','Charlotte','chemical','chemist','chessboard','chicken','chief','child','china','Chloe','Christopher','citizen','Claire','clean','cleaned','clock','closet','cloth','cloud','coach','cobra','cockroach','coffin','coin','collapse','collapsed','computer','condo','confess','confessed','consumer','container','cook','cooked','cookie','corner','corpse','cot','couch','counter','cow','crack','cracker','crate','crave','craved','crawl','crayon','creature','crib','cried','crocodile','crown','crumple','crumpled','cry','crystal','cubicle','cup','cupboard','cushion','customer','cylinder','dance','danced','Daniel','David','dealer','deck','declare','declared','decompose','decomposed','deer','desk','device','director','discover','discovered','dish','disintegrate','disintegrated','doctor','dog','doll','donkey','donut','double','doubled','dragon','draw','drawer','drawn','dream','dreamed','drew','drink','driver','duck','duke','dumpster','dungeon','dust','dusted','Dylan','e','eat','eaten','Eleanor','Elijah','Elizabeth','Ella','Ellie','Emily','Emma','enjoy','enjoyed','enlarge','enlarged','envelope','Ethan','Evelyn','examine','examined','expect','expected','farmer','father','fed','feed','fig','find','fish','fishbowl','flag','float','floated','floor','flower','fly','foreigner','forward','forwarded','found','fox','freeze','fridge','friend','frog','frown','frowned','froze','frozen','fruit','futon','Gabriel','game','garden','gasp','gasped','gave','genius','giant','giggle','giggled','giraffe','girl','give','given','glacier','glass','glue','goose','governor','Grace','gravel','Grayson','grew','grow','grown','guard','guest','guitar','gumball','guy','hamburger','hammer','hammock','hand','handed','hanger','Hannah','Harper','hat','hate','hated','haystack','Hazel','headmaster','hear','heard','hedge','hedgehog','held','helicopter','help','helped','hen','Henry','hero','hippo','hold','hole','hope','hoped','horse','host','house','human','hunt','hunted','imagine','imagined','improve','improved','in','inflate','inflated','intend','intended','investigate','investigated','Isaac','Isabella','itch','itched','Jack','jacket','Jackson','Jacob','James','jar','Jaxon','Jayden','jeep','jigsaw','jog','jogged','John','Joseph','Joshua','journalist','judge','juggle','juggled','Julian','kennel','key','keyboard','kid','king','kitty','knew','knife','know','known','ladder','lamb','LAMBDA','lamp','landlord','laugh','laughed','lawyer','Layla','leaf','leaflet','Leah','lemon','lend','lended','Leo','Levi','Liam','like','liked','Lillian','Lily','Lina','Lincoln','lion','liver','loan','loaned','log','Logan','lollipop','long','longed','love','loved','Lucas','Luke','Luna','machine','Madison','mail','mailed','manager','mandarin','Mason','Mateo','Matthew','mattress','mean','meant','melon','Mia','Michael','microwave','Mila','mirror','miss','missed','molecule','monk','monkey','monster','moose','mother','mound','mouse','muffin','mug','murderer','nail','nap','napkin','napped','Natalie','Nathan','need','needed','needle','nest','newspaper','nmod','Noah','Nora','notebook','notice','noticed','nurse','nursed','observe','observed','offer','offered','Oliver','Olivia','on','Owen','pack','package','packed','pad','paint','painted','painting','palace','pancake','panel','paper','parcel','pass','passed','passenger','patient','Paula','pedestal','pen','pencil','Penelope','penguin','penny','penthouse','pepper','philosopher','piano','pickle','pierce','pierced','pig','pile','pillar','pillow','pit','pizza','plan','planned','plant','plaque','plate','platter','pod','podium','poet','poke','poked','politician','pony','pool','post','posted','poster','pot','potato','prefer','preferred','present','president','pretzel','priest','prince','princess','prisoner','producer','professor','promise','promised','prove','proved','puddle','pumpkin','pupil','puppy','purse','pyramid','queen','rack','radio','rag','raisin','ran','read','recipient','redden','reddened','refrigerator','rent','rented','researcher','resident','respect','respected','return','returned','Riley','ring','road','rock','rod','roll','rolled','room','rose','rug','run','Ryan','sack','said','sailor','Samuel','sandwich','saucepan','Savannah','saw','say','scarf','Scarlett','scientist','scoff','scoffed','scream','screamed','sculpture','seat','Sebastian','see','seed','seen','sell','send','sent','servant','serve','served','shark','shatter','shattered','sheep','sheet','shelf','shell','ship','shipped','shirt','shoe','shoebox','shorten','shortened','sink','sketch','sketched','skull','Skylar','sleep','slept','slid','slide','slip','slipped','smile','smiled','smirk','smirked','snap','snapped','sneeze','sneezed','snooze','snoozed','snore','snored','soap','sock','sofa','Sofia','sold','soldier','Sophia','soup','spaceship','speaker','sphere','split','spokesman','spoon','squeeze','squeezed','squirrel','stab','stabbed','stack','stadium','stage','stand','statue','Stella','stool','storage','strawberry','student','studied','study','stutter','stuttered','support','supported','surface','surgeon','swamp','sweetcorn','sword','table','tabletop','talk','talked','taxi','teacher','teapot','teleport','teleported','tenant','tent','that','the','The','theme','Theodore','think','Thomas','thought','threw','throne','throw','thrown','tiger','tin','to','tolerate','tolerated','tomb','tool','toothbrush','torch','toss','tossed','touch','touched','tourist','towel','tower','toy','trailer','train','trainee','trampoline','trap','tray','tree','tried','tripod','trophy','truck','trunk','try','tub','tube','turkey','turntable','turtle','tv','TV','value','valued','valve','van','vase','vehicle','vessel','Victoria','Violet','visitor','wagon','walk','walked','want','wanted','wardrobe','warrior','was','watch','weapon','well','whale','William','windowsill','wine','wire','wired','wish','wished','wolf','worm','worship','worshipped','writer','Wyatt','yacht','yearn','yearned','yogurt','zebra','Zoe','Zoey'])
def replace_out_of_vocab_words_with_unknown_token_and_remove_period(sentence):
  sentence = sentence.replace(" .", "").replace(".", "")
  words = sentence.split(" ")
  return " ".join([word if word in main_train_vocab else "unknown" for word in words])
# e.g. transforms ['A rose was helped by a dog .', 'William painted Ying .']
# into ['A rose was helped by a dog', 'William painted unknown']
# (where "A rose was helped by a dog ." is the first line in the official COGS examples, so all in training vocabulary; "William painted Ying" is an example of mixed in-vocabulary/out-of-vocabulary content chosen as the author is named William and their partner is named Ying ( https://github.com/i-am-ying-li ))
sentences = [replace_out_of_vocab_words_with_unknown_token_and_remove_period(sentence) for sentence in sentences]

lfs_computed = []
exact_matches = []
for idx in range(len(sentences)):
  try:
    suppress_output = len(sentences)>1000
    output = process_example(sentences[idx], suppress_output)
  except:
    print(f"Could not process input '{sentences[idx]}'")
    output = ""
  lfs_computed.append(output)
  # in COGS "semantic exact match" reorderings and consistent variable index substitutions that do not change the semantics are still correct, and that is our metric to report per Wu et al 2023, for ReCOGS (we also check exact match there too)
  # but here we just check exact match
  exact_matches.append(1.0 if output.strip() == lfs_true[idx].strip() else 0.0)
  mean_em_score = np.array(exact_matches).mean()
  num_em_right = np.array(exact_matches).sum()
  em_ci_pct = get_clopper_pearson_confidence_interval(len(lfs_computed), num_em_right)*100.0
  print_stdout_and_file(f"Exact match score on {pos_desc} {len(exact_matches)} of COGS {label}:\n{disclaimer}\n{mean_em_score*100:0.2f}% or {num_em_right} out of {len(exact_matches)} (95% confidence interval: {em_ci_pct[0]:0.2f}% to {em_ci_pct[1]:0.2f}%)", running_scores_logfile_handle)
  if args.use_gen_split:
        gen_em_df = pd.DataFrame([{"Exact Match": exact_matches[jdx], "Category": labels[jdx]} for jdx in range(idx+1)], columns=["Exact Match", "Category"])
        print_stdout_and_file("\n", running_scores_logfile_handle)
        print_stdout_and_file(f"Exact Match % by category:", running_scores_logfile_handle)
        em_by_category_with_ci = get_percentages_with_ci_groupby_binary_data(gen_em_df, "Category", print_result=True, file_handle=running_scores_logfile_handle)
        print_stdout_and_file("\n", running_scores_logfile_handle)
  if idx % 10 == 0:
    # update CSV (these are small so can rewrite each time)
    output_df = pd.DataFrame([{"Input Sentence": sentences[jdx], "Logical Form Predicted": lfs_computed[jdx], "Label": labels[jdx]} for jdx in range(idx+1)], columns=["Input Sentence", "Logical Form Predicted","Label"])
    output_df.to_csv("lf_output.tsv", index=False, sep="	")

output_df = pd.DataFrame([{"Input Sentence": sentences[jdx], "Logical Form Predicted": lfs_computed[jdx]} for jdx in range(len(lfs_computed))], columns=["Input Sentence", "Logical Form Predicted"])
output_df.to_csv("lf_output.tsv", index=False, sep="	")

exact_matches = [1.0 if lfs_computed[idx].strip() == lfs_true[idx].strip() else 0.0 for idx in range(len(lfs_computed))]
mean_em_score = np.array(exact_matches).mean()
num_em_right = np.array(exact_matches).sum()
em_ci_pct = get_clopper_pearson_confidence_interval(len(lfs_computed), num_em_right)*100.0

print_stdout_and_file("\n\n\n", running_scores_logfile_handle)
print_stdout_and_file(f"Exact Match score on {pos_desc} {len(sentences)} of COGS {label}:\n{disclaimer}\n{mean_em_score*100}% or {num_em_right} out of {len(sentences)} (95% confidence interval: {em_ci_pct[0]:0.2f}% to {em_ci_pct[1]:0.2f}%)", running_scores_logfile_handle)
print_stdout_and_file("\n\n\n", running_scores_logfile_handle)

running_scores_logfile_handle.close()

# quit RASP
input_lines = [bytes("quit()\n", 'utf8')]
stdin_handle.writelines(input_lines)
stdin_handle.flush()
stdin_handle.close()
stdout_handle.close()
proc.kill()
