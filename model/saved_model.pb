??'
?&?&
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
K
Bincount
arr
size
weights"T	
bins"T"
Ttype:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Cumsum
x"T
axis"Tidx
out"T"
	exclusivebool( "
reversebool( " 
Ttype:
2	"
Tidxtype0:
2	
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(?
=
Greater
x"T
y"T
z
"
Ttype:
2	
?
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?
.
Identity

input"T
output"T"	
Ttype
l
LookupTableExportV2
table_handle
keys"Tkeys
values"Tvalues"
Tkeystype"
Tvaluestype?
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype?
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype?
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
?
MutableHashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
?
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
RaggedTensorToTensor
shape"Tshape
values"T
default_value"T:
row_partition_tensors"Tindex*num_row_partition_tensors
result"T"	
Ttype"
Tindextype:
2	"
Tshapetype:
2	"$
num_row_partition_tensorsint(0"#
row_partition_typeslist(string)
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
m
StaticRegexReplace	
input

output"
patternstring"
rewritestring"
replace_globalbool(
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
StringLower	
input

output"
encodingstring 
e
StringSplitV2	
input
sep
indices	

values	
shape	"
maxsplitint?????????
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.9.12v2.9.0-18-gd8ce9f9c3018??%
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

: *
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
: *
dtype0
?
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

:  *
dtype0
?
Adam/embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?N *,
shared_nameAdam/embedding/embeddings/v
?
/Adam/embedding/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding/embeddings/v*
_output_shapes
:	?N *
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

: *
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
: *
dtype0
?
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

:  *
dtype0
?
Adam/embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?N *,
shared_nameAdam/embedding/embeddings/m
?
/Adam/embedding/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding/embeddings/m*
_output_shapes
:	?N *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
|
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	table_7*
value_dtype0	
l

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name3447*
value_dtype0	
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

: *
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
: *
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:  *
dtype0
?
embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?N *%
shared_nameembedding/embeddings
~
(embedding/embeddings/Read/ReadVariableOpReadVariableOpembedding/embeddings*
_output_shapes
:	?N *
dtype0
G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R
H
Const_1Const*
_output_shapes
: *
dtype0*
valueB B 
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R 
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R 
??
Const_4Const*
_output_shapes	
:?N*
dtype0*??
value??B???NBmovieBfilmBoneBlikeBgoodBevenBitBwouldBtimeBreallyBseeBstoryBwellBmuchBgetBbadBgreatBalsoBpeopleBfirstBmoviesBmadeBfilmsBwayBmakeBcouldB
charactersBthinkBthisBwatchBtheBmanyBseenBtwoB	characterBneverBloveBactingBplotBbestBknowBlittleBiBshowBlifeBeverBbetterBendBstillBsceneBsayBmanBscenesB	somethingBgoBimBbackBwatchingBrealBthingBactorsByearsBthoughBfunnyBanotherBactuallyBmakesBnothingBworkBfindBlookBgoingBlotBnewBeveryBoldBusBpartBdirectorBcantBthatsBthingsBwantBcastBquiteBprettyBseemsBaroundBworldByoungBgotBtakeBhoweverBfactBenoughBhorrorBthoughtBbigBgiveBiveBmayBlongBthatBwithoutBallBrightBmusicBsawBalwaysBtimesBgetsBoriginalBseriesBandBcomedyBcomeBalmostBroleBmustBinterestingBtheresBwholeBleastBguyBpointBactionBdoneBsBbitBminutesBscriptBfarBmeBmightBanythingBhesBsinceBfeelBlastBisBfamilyBprobablyBperformanceBkindBawayBtvByetBworstBfunBratherBsureBanyoneBfoundBplayedBgirlBmakingBthemBbelieveBwomanBalthoughBtryingBshowsBcourseB
especiallyBhimBdayBhardBcomesB
everythingBgoesBputBdvdBendingBworthB	differentBplaceBmaybeBlookingBmainBscreenBbookBsenseBhereBlooksBreasonBwatchedBthreeBsetBtrueBmoneyBeffectsBjobBsomeoneBtogetherBplayBplaysBactorBsaidBinsteadBeveryoneBlaterBseemBtakesBagainBaudienceBamericanBspecialB	beautifulBseeingBwarBleftB	excellentBoutBjohnBversionBideaBnightByouBaBinBshotBblackBsimplyBfanBhighBelseBhouseB
completelyBniceBdeathBusedBonBpoorBkidsBwifeBreadBalongBfriendsBshortBhelpByearBsecondBmindBthereBeitherBstarBhomeBmenBboringBlessBenjoyBtryBgivenBuseBneedBhalfBclassicBwrongBrestBperformancesBtrulyBnextBdeadB
productionBstupidBlineB	recommendB	hollywoodBfatherBcoupleBstartBtellBwomenBletBcameBawfulBothersBherBgettingBrememberB
understandBterribleBfullBschoolBcameraBmomentsBperhapsBkeepBmeanBsexB	wonderfulBplayingBnameBvideoBepisodeBbudgetB
definitelyBoftenBhumanBnotBpersonBperfectBgivesBtopBsoBtooBfaceBearlyBupBsmallBlinesBdialogueBwentBpieceBguysBstarsB
absolutelyBfinallyBcaseB	certainlyBtitleBheadBbecomeBlikedBlovedBhopeBworseBentertainingBsortBlostByesBstyleBfeltBmotherBentireBpictureBsupposedBoverallBwrittenBseveralBliveBlaughBboyBcinemaBproblemBwasteBfriendBsoundB	beginningBohBmrBbasedBtotallyBfansBcareBwantedBseemedBdarkBnowBhumorBidB	directionB
throughoutBdespiteBfinalBchildrenBlivesBexampleBguessBalreadyBleadBbecomesBevilBgameBdramaBdaysBturnBcalledBunfortunatelyBlowBableBwhiteBgirlsBwantsBhistoryBfineBqualityBhorribleBamazingBBbutBwritingBworksBtheyreBactBkillerBkillBtriesBflickBmichaelBenjoyedBturnsBgaveBpastBsideBpartsBsonBmatterBfavoriteB	brilliantBstuffBeyesBbehindBexpectBonesBcarBtownB	obviouslyBrunBstartsBviewerB	sometimesBkilledBsoonBdirectedBlateBetcBgenreBheartBthinkingBasBsaysBartBcityBactressBwithBgroupBtookBdecentBhighlyBillBheardBhappensBhellBfeelingBkidB
experienceBchildBanywayBexceptBfightBhourBwasBdoBleaveBbloodBmomentB	extremelyBcloseBtoldBstoriesBtoBcomingBpoliceBcannotBlackBchanceBscoreBinvolvedBhandBhappenBsaveB	hilariousBwonderBrolesBlookedBstrongBokBdaughterBtypeBgodBlivingBviolenceBforBparticularlyBhappenedBattemptBcompleteB	includingBobviousBshownBseriousBsimpleBtakenBstopBjamesBmurderBcoolBbeBagoBnoBoffBpleaseBvoiceBsongBrobertBcrapBacrossBdavidBletsBmoreBknownBexactlyBrealityBopeningBreleasedB	seriouslyBpossibleBtodayBnumberBinterestBsayingBusuallyBjokesBhoursBaloneBslowBnoneBsadBhugeBcinematographyBwishBrelationshipBenglishBorderBcareerBannoyingBwhatBtalentBwhoseBageBhitBshotsBgoreBofBrunningBcutBendsBwhoB
ridiculousBstartedBmajorBcallBbrotherBheroBviewBwhichBtakingBusualBfemaleBdocumentaryBbeyondBchangeBmostlyB	importantBbodyBopinionBsomewhatBknewBpowerBknowsBsillyBwordsBdisappointedBratingB
apparentlyBscaryBturnedBfindsBnovelB	basicallyBstrangeBwordBdueBhusbandBlevelBfourB	attentionBhappyBcountryBepisodesBtalkingBuponBroomBproblemsB	directorsBclearlyBlocalBcheapB
televisionBmusicalBwhatsBifBsingleBeventsBjackBmissBsequenceBbritishBtellsBtalkBthenBreviewBmodernBlightBwhetherBsongsBpredictableBaddBdialogBfrenchBeasilyBsetsBbeforeBappearsBaboutBbringBlotsBfutureBentertainmentBorB
soundtrackBviewersBtenBgivingB	enjoyableB
supportingBsimilarBmentionBfallsBteamBearthBgeorgeBmessageBwhyBneedsBmovingB	surprisedBbunchBwithinB	storylineBromanticBfiveBspaceBshowingBhateBcertainBtriedBparentsBfilmedBclearBsequelBdullBcomicBwaysBmiddleBsorryBamongBtheaterBthemeBreleaseBthrillerBkingBeffortBotherBfallBnamedBeasyBwereBstayBcommentsBusingBtypicalBfeelsBwriterBgoneBnearBdealBkeptBbuyBdoubtBgreatestBworkingBmonsterBnearlyBelementsBavoidBsuspenseBdieBsubjectBmeansB	fantasticBeditingBactualBtaleBviewingBpointsBimagineBdrBrichardBpeterB	realisticBstraightBthBgeneralBboysBrockBladyBfamousBcheckBbroughtBminuteBokayBsisterBmysteryBfeatureBdidBclassBmoveBmaterialBleadsBforgetBformBbeginsBsomehowB
believableBreviewsBperiodBhearBdogB	animationBsurpriseBrentBfigureBlearnB
eventuallyBpremiseB	sequencesBpaulB
atmosphereBcrimeBweakBsitBeyeB
particularBexpectedBwhosBtomBindeedBfastBkillingBwaitB	situationBshameBfollowBdecidedBredBtruthByorkBdeepB	difficultBlameBwhateverBaverageBleavesBseasonBpoorlyBbecameBpossiblyB	emotionalBfootageBneededBforcedBdanceB	memorableBnoteBmeetBbeginBcreditsBstandBreadingBoscarBscifiB	otherwiseBopenBcrewBunlessBromanceBquestionBthirdBwritersBsexualBimdbBfeaturesBdownBbeautyBdoctorB
filmmakersBsuperbBsocietyBwriteB
interestedBmeetsBhandsBnatureBpreviousBcheesyB	perfectlyBhotBgayBmasterpieceB
screenplayBitsBeffectBwesternBpersonalBinsideBcommentBlaughsBtotalBsoundsBplusBkeepsBquicklyBtowardsBareBislandBresultBweirdBmaleBboxBstreetBbB
incrediblyBjapaneseBearlierBdeBbattleBbadlyBamericaBworkedB
backgroundBrealizeBuniqueBcopyBairBcrazyBmessBsettingBolderBvariousBstageB	followingBfreeBhimselfBplentyBpowerfulBmarkBbringsBappearBbusinessBleeBaskBadmitBdevelopmentBfairlyBjoeBcreepyBapartBactedB	directingB	portrayedBspentBreasonsBfrontBdumbBbabyBremakeBjokeBforwardBleadingBdramaticBoutsideBrichBbillBfireBdeservesBwastedBpayBfailsBwaterBsuccessBmeantBattemptsBideasBmanagesB	expectingBreturnBlaB	politicalBmissingBcreateBagreeBwilliamBcoverBpartyBrecentlyBcaughtBfightingBhardlyBtwistBcopBpresentBmatchB
girlfriendBunlikeBsecretBdreamBcleverBmembersBitselfBendedBlargeBbrothersBlaughingBcuteBspoilersBplainBescapeBtellingBoddBtalentedBbreakB	potentialBzombieBmissedBpureBgermanBpaceBdisneyBcreatedBeraBwaitingBcastingBgunBmarriedBnudityBvillainBwhenBpublicB
incredibleBoverBvisualBholdBcauseBconsideringBseesBslightlyBlistBvanBsadlyBdiedBitalianBatBcartoonBspeakBfamiliarBstateBusesBfantasyBdecidesB	mentionedBcompanyBproducedBentirelyBtrainBpopularBneitherBcreditB
convincingBwroteBofficeBfollowsBboredBflatBtensionBcomparedBrateB
appreciateB	portrayalBvalueBtroubleBstoreBfitBamountBchoiceBsuddenlyBformerBmovesBlanguageBbiggestByoungerB	audiencesBintelligentB	producersBcommonBimagesBdancingBsweetBscienceBcenturyBaliveBrecommendedB
successfulB
situationsBviolentBforceBprojectBconceptBfearBsocialBlongerBcollegeBamusingBfilledBcoldBhairBdontBspendB	pointlessBexcitingB	christmasBmadBkillsB
ultimatelyBpatheticBdecideBbizarreBbandBscottBrecentBsickBbasicBpositiveBfocusBconsiderB	effectiveBdepthBthroughBlikesB	questionsBspiritBbooksByeahBcontrolBawesomeBasideBjaneB
impressiveBchangedBsolidBfakeB	involvingBbarelyB
impossibleBhumourBgarbageBshowedBsingingBhonestBrevengeBrespectBvaluesBsoldiersBfromBleavingBaccentBstudioBmasterB	somewhereBgeniusBalienBmeaningBimmediatelyBwalkB	literallyBfailedB	chemistryBchannelB
consideredBboughtBthanksBcharlesBbrainBcultureBtrashBshootingBprisonBbenBhonestlyB
conclusionBadultBabilityBgangB
disturbingBpickBfairBtouchBsittingBsurprisinglyBgladBthinksBsouthBroadBaddedBtripBstarringBrunsBexplainBcampBwinBnormalBmoodBjourneyBbotherBvampireBwestBissuesBarmyBcatBtwistsBveryBfictionBmyB	adventureB
personallyBtoneBstickBlikelyBsexyBnowhereBblueBpurposeBjimBdrugBbedBnobodyBweekBparkBbeautifullyBtasteBtermsBremainsB	generallyB	availableBtoughBsilentBsubtleBpicturesB	cinematicBaspectBchaseBwhileBanimatedBcomplexB
appearanceButterlyBshootBsamBfrankBlovesBafterBpresenceBcultBstandardB
adaptationBcharmingBnakedBmanagedBlovelyB
historicalBheBslasherBnaturalBrideBplanetBphotographyBcatchBplanBhaveBedgeBdisappointingB
constantlyBterrificBmaryBchangesBsupposeBcomediesBthrownBdoorBmagicBinnocentB
governmentBappealBsceneryBmilitaryBzombiesBemotionsBdetailsBlondonB	laughableBknowingBgiantBplacesBsoulBrareBpiecesBdateBcomputerBnamesBpassBattackBchrisBwalkingBindianBtouchingBsteveBspoilerBsmartBunbelievableB	narrativeBminorBslowlyBpainfulBstandsB
impressionBintendedBbottomBthusBheyBexcuseBcostumesBthankBselfBdadBcontainsBtrackBjonesB	presentedBfilmingBfeelingsBghostBputsBmakeupB	everybodyBcriticsBwildBfestivalBlandBfinishBbesidesBmakersBoutstandingBthrowBequallyBbriefB	detectiveBbuildingBthemesBoperaB
mysteriousBlawBparisBbondBactsBhopingBcentralBsurelyBmistakeBdoesBclimaxBstunningBaspectsBownBforeverBsentBwowBstudentsBmannerBdisappointmentBmainlyBexpectationsBraceBclubBrayBharryBvictimBfullyBnoticeBemotionBtiredBedB	boyfriendBonceBvictimsBopportunityBhurtBfascinatingBloudBjusticeBcharmBcharlieBawardBsupportBlaughedBimageBofferBproducerBhospitalBbruceBwoodsBpainBtonyBdreamsBcryB	confusingBfacesBstBfindingBaheadBincludeBcopsBstudentBsuggestBshipBrandomB	happeningBconfusedBgreenBchristopherB
themselvesBspeakingBfreshB	exceptionBmansBnewsBflicksBcontentBmillionBlivedBbossBholesBfellBtwiceBanswerBfolksBmotionBdiesBcolorBsmithB
differenceBratedBrelationshipsBlocationBdrugsBgradeB
compellingBpullBlacksBsummerB	seeminglyBgemBmarriageB
supposedlyB	impressedBfallingBheavyBextremeBlikableBaffairBhelpsBapproachBagentB	developedBaddsBlightingBbatmanB
collectionBshockBbillyBinformationBfunniestBshareBradioBmoralBmartinBbornBappearedBdeliversBcreativeBpaidBiiBhowBindustryB	wonderingBflawsBbarBputtingBplaneBgorgeousBdetailBtrailerBintenseBrussianB	thereforeBelementBdriveBfollowedBadultsBrentedBoffersBallowedBteenBremindedBmerelyBvsBsystemBsixBledBkeyBuglyBanymoreB	forgottenBanimalsBgamesBeventB	americansBmediocreBprovidesBimpactBmixB
attractiveBmsB
filmmakingBabsoluteBnegativeBbecomingBbeatBstuckBdrawnBsuicideB	standardsBgroundBmachineBontoBshockingBloverBcountBareaBserialBsecondsBhotelBpornBimaginationBchurchBcaptainBliesB	christianBstatesBgraceBsuperBspotBreadyBdamnBdirtyBcompareBclichéBworthyB
whatsoeverBpickedBmomBhelpedBdisasterBpersonalityBtragicBreturnsBmurdersBstationBdeliverBangryBturningBstewartB
thoroughlyBsoldierBnastyBtragedyBfellowBfashionBphoneBlatterBartisticBteenageBmovedBcgiBactionsBcreatureB
commentaryBprovesBepicBwoodenBprocessBtortureBquickBadditionBloseBkevinBdirectB	childhoodBafraidBsearchBpassionB	actressesBsevenBwearingBmemberBlistenBinspiredBcarryBflyingBcontinueBbyBaskedB	scientistBbeganBaccidentBwillingBtheatreBprovideBkellyB	favouriteBdyingBjasonBdescribeB	apartmentBwantingBheldBbitsBwarsBteacherBqueenBseaBrealizedBprofessionalBextraBtodaysBrarelyBnumbersBclichésBmonstersB
introducedBsucksBpleasureB	filmmakerBphysicalBmemoryBintelligenceBhenryBclothesBzeroBstruggleBsleepB	redeemingBfoodBenergyBacceptBjustBtimB	necessaryBjudgeBindependentBhiddenBwilliamsBwallBsightBallowBnoirBbloodyBsuspectBpopB	watchableBthomasB	dangerousBasksBanywhereB
intriguingBdesignBtearsBartsBstepB
surprisingBholdsBcallsBmouthBjerryBdeserveBsuperiorBryanBchineseBfatBcomedicBblameBdeeplyBanimalBtrustBwonderfullyBremindsBaccurateBsatBincludesB	religiousBdoubleBrapeBmartialBheroesBexplanationBunderstandingBfoxBanybodyBwarningBunknownBplotsBhorseBsomeBdannyBthinBlimitedBapparentBunusualBtreatBnicelyBgoldBwoodBdesertBbrutalBieBhatedB
remarkableBpacingBjoyB
friendshipBstephenBweBmoonBmineBlordBmikeBloversBcriminalBwhilstBmonthsBmemoriesBheadsBgagsBcBoppositeBjrBlooseBjohnnyBcapturedBringBgrandBdrivingB	desperateBnoticedBmetBvillainsBpowersBmyselfB	locationsBunnecessaryBscaredBbuildBvhsB	communityBvisionBpageBlearnedBleaderBjeanB	knowledgeBengagingBacademyBstockBmentalBhumansBdiscoverBbrianBalBuncleB
referencesBissueBcanBjacksonBgoldenBboatBalanBresponsibleBsomebodyBplayersBofficerBstartingBnormallyBwindBordinaryBcreatingBboardB	accordingBsmileBconstantBarthurBcarsBartistBkeepingBtreatedBpriceBanB	technicalBmissionB	explainedB
originallyBmanageBhitsBabsurdBplayerBnationalBlackingBgrowingBfloorBfinishedBconflictBskipBsistersBmagnificentB	lowbudgetB
connectionBrollBmrsBeddieBwindowBrecordBcurrentB
comparisonBanneBsurviveBhumanityBfightsBrealismBproveBoccasionallyBkillersBfieldBexistBeffortsBmurderedB
generationBdeservedBrubbishBmorningBlengthBfordBwittyB	naturallyBlovingBheresBforcesBsameBlegendBgottenBfailBnumerousBincludedBcaptureBbiggerB	hitchcockBfaithBcrappyBbehaviorBsavingBbobBseanBrobinBlossBfinestBbrownBgenuineBdressedBblandBshopBhopesBbrightBnonsenseBinstanceBtheyBprivateBdesireBbrokenBwheneverBsingerBpilotBhowardBdealsBtapeBhumorousBfBcutsBadviceByourselfBrevealedBessentiallyBeatBmixedBgunsBterriblyBmediaBkindaBtowardB	teenagersBstoneBforeignBdealingBweveBunfunnyBsoapBpulledBenglandBbuddyBwitchBgaryBawareBstereotypesBfranklyBresultsBemptyBstreetsBrelateBregularBquietBjenniferB	featuringB	concernedBpsychologicalBpairBjumpBcastleBcableBeuropeanB
discoveredBdavisBblindBsuchBskillsBsheerB	reviewersBpostB	meanwhileBjeffBfinaleBfeetBpriestBmaxBinsaneBsuitBspanishBfavorBtexasBsavedBriverBjoanBhigherBaliensBtalentsBpartnerBinternationalB	genuinelyBvillageBpeoplesBfateB
underratedBsignB	nightmareBcrossBscreamBmooreBattitudeBvampiresBreceivedByouthButterBsuckedB	presidentBmilesBheroineBcontextBwinningBstudyBprogramBvisitBsingBrangeBanthonyBunableBportrayBdrunkBvoteB
rememberedBfameBtwentyB	screamingBreachBhadBdatedBnickB
flashbacksB	continuesBbreaksBworldsBspectacularBstealB
irritatingBfailureBevidenceBdreadfulBvisualsBunrealisticBopensBluckyBlevelsBdebutBcrowdBclassicsBreactionBownerBlessonBeditedBultimateBawkwardBstrengthBallenBweddingBsatireBkBgrowB	describedBprotagonistBladiesB	creaturesBrBheavenBballB
unexpectedBtravelBdogsBadamBstevenBshallowBranBframeBcameoBfootballBdecisionB
australianBtaylorBsiteBpretentiousBcornyBidentityBdecadeBlogicBhunterB	existenceB	contrivedBperspectiveBfreedomBfaultBdBparodyBfranceB	discoversBcuriousBtheyveBluckBjimmyBgoryB
commercialBvisuallyBsellBprovidedBnativeBgonnaBflyBandyB
delightfulBchooseBkickBinvolvesBiceB	deliveredBgrewBendlessBbombBsakeBericBcombinationBwideBtinyBtheatersBseatBrescueBmeetingBlewisBheckBcostBreliefBcaresBbrilliantlyBlaughterBgraphicBdouglasBcreatesBagesBrulesBdragBbeachBbringingBaskingBpromiseB	halloweenBbankBtypesBrecallB
individualBfillBeatingBcapableBbarbaraBallowsBakaBteenagerBtalksBstandingBremainBprovedBproductBlosesBformulaBemotionallyBbodiesBassumeBtrekBgangsterBeuropeB
disgustingBfamiliesBdestroyBtraditionalBsympatheticBjulieBcontrastBshakespeareBnorthBjesusBbelievesBasianBspeechBroundBneverthelessBmajorityBdevelopBdanBenemyBdickBcleanBoverlyBfredBcenterBbothB	treatmentBfactsBchickBwiseBtestBriseBreviewerBportraysBonlyBcrashBcashBbuiltBtediousBstronglyB	hopefullyBfathersBexecutedBembarrassingBdollarsBauthorBweeksBviewedBstealsBhorriblyBwillBsheriffBancientB	subtitlesBsequelsBrobBcorrectBaliceBunitedBsurrealBsuddenBproperBalexBmattersBmattBjapanBcostsBbbcBhandsomeBgeneBcandyBwalkedBmostBdidntBawardsBthoughtsBterrorBresearchBcircumstancesBcausedBspeedBlousyBblondeBwalksBvoicesBrentalBhallBclueBukBstudiosBskyBkongBexcitedBannBshockedBnetworkBmodelBextrasBentertainedBwayneB
theatricalB	sufferingBpityBeveningBdubbedB	slapstickBpassedBlosingB
gratuitousB
depressingBcoreBcartoonsBsurfaceBinsultBstoppedBsourceB
portrayingBdevilBroseBreligionBlouisBlargelyBenterBcontemporaryB
technologyBproduceBexperiencedBdriverBdarknessBbeingBvehicleBtwistedBsoftBpleasantBhearingBbetBbearB	committedBchosenB
amateurishBchiefBassBtillBsendB	painfullyBlarryBteensBspoilBprinceBplansBmaskBversionsBuniverseBruinedB	qualitiesBlearnsBfactorBwitB
propagandaBmarryBjackieBfitsB	virtuallyBsaturdayBpaperBgermanyBfleshBannaBharrisBdepictedBsharpBpatrickBbmovieBwitnessB	universalBspendsBbecauseBsegmentBlakeBexploitationBangelBalbertBwerewolfBherselfBdennisBanglesBserviceBlynchBlesbianBlatestB	influenceBhauntingBwhereB
relativelyBhuntBgrownBsportsBjungleBasleepBvarietyBtendB	substanceB	recognizeB	promisingBpracticallyBmorganBdemonsBblowBtableBskinBsheBmindsB	daughtersBscareBholdingBconvinceB
worthwhileBsympathyBseasonsBguiltyBexperiencesB
excitementBclaimBappropriateBveteranB
satisfyingB	performedB	nominatedBdryB
departmentBturkeyBtargetBhalfwayBdisplayBcapturesBtrainingBedwardBdangerB
previouslyBcanadianBcallingBunfortunateBspeaksBjosephBtruckBsuffersBmovementBholmesBfootBextraordinaryBconversationBwinnerBrogerBkeatonBfridayBbelievedBtarzanBpsychoB	professorBpeaceBjewishBhideBfeaturedBcostumeBpresentsBhillB
washingtonBrobotBmidBgordonB	appealingBaccidentallyBobsessedBidiotBhandledBgrantedB
continuityBcageBxB	structureBroutineBharshBhangingBclichédB	childrensBsuspensefulBsupernaturalB	encounterBeightBdonaldBrobertsBcategoryBamateurBvietnamBdegreeBdeadlyBaccentsB	surprisesB	statementB	offensiveBdrawBchoseBbotheredBstorytellingBscaleBhauntedBdirectlyBbroadwayBabuseBsafeBclaimsBbreakingB
adventuresBwesternsBwarmBrevealB
refreshingB	favoritesBcruelBaintBmountainBmassiveBforgettableBfiguredBdeathsBweekendBtrappedBseagalBregretBpraiseBlaneBinitialBdropBcryingBtouchesBprimeBpatientBmarketBirishBtrilogyB
overthetopBexactB	elizabethBdudeBunlikelyB
thankfullyBstereotypicalBsirBservesBlightsBforgotB
everywhereB	destroyedBcuttingBwelcomeBlowerBandorBrealizesBpriorBpoliticsB
mainstreamBkateBdeanBcloserB	amazinglyBwalterBsectionBrequiredBforthBcoveredBpreparedBnaiveB	melodramaBkissBsidesBruleBpullsBjunkB	interviewBeffectivelyBbeliefBanimeB	abandonedBsecurityBproudBafricaBfishBexpressBcolorsBsummaryBnurseBinsightB
californiaBsortsBsonsBroyB	legendaryBhitlerB	highlightBfrighteningBalrightBwhoeverBstolenBproductionsB
interviewsBgrittyBuBspoofBreporterBremotelyB	flashbackBafricanBwearB
reputationBprintBpreferB	executionBrepeatedBrentingBparkerBgraveBusaBurbanBripBfareBsimonBpositionBnonethelessBmildlyBmexicanBgreaterBchinaBartistsBarmsBaccountBwilsonBweaponsBpacedB	narrationBmultipleBlifetimeBjuliaBinnerBhiredBgrantBfoolBfocusedBflightBfalseBbrooksBvictorBsunBsouthernBquirkyBmistakesB	downrightBcriticalBandersonBstatusBrightsB
occasionalBdemonBchaplinBwouldveBviaBmagicalBforestBexplainsBdeeperB	convincedBbeenBundergroundBsoundedBrussellB	regardingBpileBisntBcellBarrivesBtechnicallyBmstB	listeningBjohnsonBhatBeastBcombinedByearoldBruinBraisedBpathB
miniseriesBimageryBdrivenB	depictionBblockbusterBtedBtaskBsupermanBstrangerBstayedBroughBripoffBplacedB
expressionBtheoryBheavilyBdeviceBclosingB	atrociousBshortsBpayingBnecessarilyBjailBhelenBfunnierBfocusesBchangingBfiguresBenvironmentBdesignedBdecadesBchoicesBbusBtightBserveBofferedBexamplesBdanielBamazedBdrewBdozenBtreasureB	sensitiveBrevealsBmassacreBlieBdraculaB
understoodBthrowsBspiteBeroticBcurseBangerBteethBscenarioBprincessBmickeyBhisBbuyingBwinsBwarnedBrainBhedBdressBcampyBbirthBuninterestingBthrowingBsuckBseekBoriginalityBkindsBkimBcarriesBblownBunconvincingBsubplotBlonelyB	initiallyBignoreB	screeningBpassingBmurphyBmurdererBmatureBlauraBindiaBchanBsitcomBsignificantBjB	entertainB	criticismBcouldveBbeastBbaseballBtermBnovelsBindieBcomplicatedBcivilBbraveBsucceedsBskillBrollingBmereBholeBflowBdinnerBaddingBwriterdirectorBtimingBchargeBblahBscaresBfacialBeverydayBdeliveryBcausesBboreBbaseBviewsBsundayBpregnantBjobsBgoofyBangleBservedBlawyerBineptBhandleBfuBbugsBamongstBsufferBstaysBspyBshadowBsarahBracismBnudeBlukeBkaneBjoinBdesperatelyBbarryB
attemptingBachieveB
strugglingBrawBnuclearBmobBsusanBnightsB	ludicrousBdescriptionBwwiiBratingsBracistBgasBfbiBconcertBbelovedBanswersBtouchedB	stupidityBringsBreminiscentBpurelyBpaintBmetalBinspirationBfaithfulBcodeBwellesBsleepingBremoteBlearningBhelpingBgruesomeBdrinkingBcontainBwolfBvacationBcrudeBcarriedBproperlyBdollarB	disbeliefBdailyBbettyBappreciatedBpsBprotagonistsBnonexistentBintroductionBhongBguestBformatBseparateBquestBperformBinternetBextentBegBdrivesBdrinkBcomicalBtommyBscriptsBmedicalBleagueBhostBfourthBstarredBexpressionsBculturalBacceptedBoliverBenjoyingBcourtBattacksB	assistantB	afternoonBtreeBtitlesBretardedB	obnoxiousBmadnessB
scientistsBsantaBrelatedBnBmothersBlisaB	strangelyBintellectualBhenceBhasBcowboyBcontroversialB	bollywoodBwearsB	referenceBbasisBtalesBnotableBinterpretationBhoodB	happinessBcontactBnowadaysBleBflawedBarmB	thousandsBshutBfabulousB
experimentBentryB
determinedB	criminalsBvonBsettingsB
revolutionBpunchBbreathtakingBpicksBcredibleBcomedianBprotectBmgmBjonBcaringBsinisterBpgBmichelleBlazyB	innocenceBcredibilityBcheeseBblairBwhereasBghostsB	fictionalBwoodyBwontBvincentBvideosBsumBstomachBronBrevolvesBexistsBescapesB	catherineBcarryingBbreathBattackedBashamedBalbeitBwarnerBmentallyBletterBgraphicsBtenseB	techniqueBsanBriskBparBgreatlyBgodsBglassBdearB	confusionBchuckBtopicB	strugglesBreplacedBlettingBfortunatelyBcolumboBchillingBbalanceBangelsBunionBstepsB
regardlessBmindlessBlosBdisagreeBdelightB	authenticBupsetB	traditionBstealingBsoldBsleazyBpoignantBironicBguardBfallenB	countriesBconveyB	challengeBbagBwealthyBsucceedBstormBmillerBlloydBlessonsBhorrificBgrippingBscreenwriterBregardBopenedBmatrixBhonorBhoffmanB
frequentlyBweaponBstopsBmanagerBidioticBflawBdareBcousinBclarkBmexicoBguessingBexceptionalBlovableBheartsBembarrassedBdoingBchairBcasesB	advantageBsocalledBitalyB	intensityB	franchiseBcynicalBtuneBswordBraiseBsinatraBmirrorBjayBgreekBdaveBclipsB	sexualityBhundredBdislikeBcooperBsuggestsBrivalBproofBpresentationBplasticBnineBmidnightBtourB	searchingBrepeatBmillionsBhintBhangBcourageBconsistsBslightBsilenceBseekingBfriendlyBfarmBcreationBboundBworryBtrickBstuntsBsnowBridingBkungBdragonBnotedB	marvelousBalasBwalkerB	obsessionBneedlessBkarenBjamieB	instantlyBgBconsequencesBairedBwastingBthousandB
techniquesBstylishBlyingB	attemptedBappearancesBtributeBreynoldsBhidingBeastwoodBcoachBbradBbitterBuselessBthiefBtextB
physicallyBjumpsBestablishedBdramasB	conditionBboredomBwarnBsufferedBspokenBsovietBsnakesBremindBoddlyBmusicalsBguideBburtBtrialBsuccessfullyBsingsB
perfectionBobjectBnaziBburnsB	appearingBtroubledBtoyBsurroundingBsavesB
performersBnazisB
meaningfulBhundredsBholidayBandrewsBunforgettableBshinesBrowBrefusesBmariaBleslieBlegsB	invisibleBbucksBbirthdayBbelongsB	australiaBhaBachievedBsallyBquoteBovercomeBninjaBhookedBexpertBcloselyBchasesBbrieflyB
attractionBteachB
sutherlandBrushBrippedBmarieBironyBgrossBgottaBgermansBgagBflashBequalB	competentBbridgeBadaptedBsendsBhardyBgloryBexerciseBbobbyBtiedBspecificBnancyB
intentionsBhorsesBgrimBfxBcheckingBbrokeBsilverBnonBjessicaBindividualsBhillsBessenceBwomansBsuitsBprofoundBlackedBcameronBcaineBspendingBsolveBnoseBimaginativeBhomageBcrimesBcornerBchasingBbusyB
uninspiredBspotsB	notoriousBmastersBfairyB	enjoymentBdawnBtornB	thrillingBpersonalitiesBmonkeyBindiansBcraftedBpoolBndBnavyBnationBmiscastB	kidnappedBkennedyBglimpseBdvdsB	connectedBcolorfulBbeingsBauntBadBtrapBshootsBrachelBpackBjonathanBincidentBgutsBfreddyBdealtBblowsBwishesBwannaBstanBfacedB	essentialBdevelopsBconcernsBatmosphericB	returningBpersonsBopposedBhB	elsewhereBbasementBanyBannoyedBagedBuncomfortableBshowerBouterBimportantlyBdialogsB
acceptableBtricksB	thrillersBsubplotsBstruckBstanwyckBsoulsBshedBshapeB
repeatedlyBpatB
outrageousBnotchBidentifyBhusbandsBcompetitionBstrikingBsetupBportraitBloneBknifeBkicksBhappilyBensembleBdestructionBwaveBstretchBshallBrushedBlesserBgrowsBdragsBdoesntBbladeBtheydB
redemptionBozBoscarsB	miserablyBinfamousBemilyBdoctorsBcreatorsB	carefullyBzoneBtalkedBsuspectsB
revelationBphilipBpetBpacinoBheistBcorruptBcameosBralphBlockedBcureBcatsB	cardboardBshadowsBnormanBneatBkoreanB
encountersBsharkB	revealingBprideBmouseBmelodramaticBlaidBignoredBhealthBfreemanB
cameraworkBweightBstuntBmeatB	godfatherBdraggedBbeerB	typicallyB	territoryBstringBstoodBsellingBpreciousBpianoB	onelinersBmelBincreasinglyBhorrorsB
highlightsBdroppedBclintB	ambitiousBadmireBwasntBvictoriaBtriteBsnakeB	remainingBrelevantBpushBpowellBmonthBmassBlibraryB
importanceBeasierBdubbingBcrisisBbuckBsophisticatedBsidneyBsandraBrogersBpulpBjerkBinvestigationB
inevitableBbibleBtoiletBresponseBmagazineB	intentionBhopedBhboBhardcoreBfosterBforgiveB	endearingBemmaB	curiosityBbrandBunintentionallyBsizeBsentimentalBreallifeBpointedBneighborhoodBlionBhuntingB
horrendousBgainB	expensiveBdianeBbreastsB	appallingBrequiresBpitchBoilBkudosBhypeBbatB	attractedBthrewBstoleBrocksBkhanBhandfulBguiltBgothicBdoorsBcommitBbritainBarnoldBaccusedBshortlyBridB
performingBminimalBlucyBjumpingBfifteenBcharismaBbattlesBbackdropBwickedB
tremendousBtravelsBstatedB
repetitiveB
reasonableBrapedBpackedBcabinBamBwatchesBwakeBtonsBthirtyBpittBmitchellBlooselyBfulciBfortuneBengagedBeerieBdancerBcommercialsBcloseupsBbrazilBargueBperBmarksB	inspectorB
explosionsB	discoveryBdigBcluesBaboveBwondersBsinB
restaurantBnoiseBloserBlindaB
admittedlyBtheseBshoesBsexuallyB	reactionsBpullingBpredatorBluckilyBlaurelBhollyBhalBdemandsB	countlessBcharacterizationBcaveBburningBbirdBangelesB
altogetherBallowingBwatersBthreatBrealiseB	possessedBoccursB
motivationBkillingsBhorridBdocumentariesBdistantBcolourBchicagoBcatholicBtallBsubB	providingBofficersBnobleBintentB
incoherentBgoalBburnBbeatingB
associatedBareasB
afterwardsBsubjectsBstaffBspoiledBsplitBoceanB
ironicallyBinteractionBelderlyBcorpseBcardB	broadcastBthrillsB
surroundedBsmokingBmessagesBheatBdefiniteB
brillianceB	alexanderB	symbolismBstanleyBsmokeBranksBposterBnotesBidiotsBfactoryBexB	believingBtheyllBsuperblyB
subsequentBsomeonesB
reasonablyBpressBpBmatthauBjakeBflynnBeveBdisplaysBcriedBburtonB
resolutionBpleasedBneckBmediumBhanksBdetailedBbearsBaudioBvagueBshyBreturnedBpickingBmansionB
hystericalBhittingBhipBglennB
frustratedBfearsBchristB	carpenterBbuttonBbeatsBarrestedBamandaBaforementionedBwallsBvirusBtradeBtorturedBspareBsegmentsB
representsBpinkBhireBgroupsBentersBemBdirectorialB
cinderellaBchancesB	celluloidBbeatenBarriveBanywaysBaccomplishedB	wrestlingBvirginBtimelessBswedishBstrikesBspookyB	spiritualBpreventBphotographedBoutcomeBmeritBfancyBevidentBescapedB	currentlyB
concerningBburiedBaimedBstripBpushedBpuppetB
presumablyB
pleasantlyBmatthewBlawrenceB	landscapeBjeremyB	insultingBhamletBgenresBfuneralBflawlessBfarceBdaringB	continuedBconneryBcharismaticBcausingBalikeBwonderedBtunesBthumbsBsurvivalB	resemblesBreceiveBprojectsBpovertyBpersonaBmakerBkingsBkenBgrandfatherBextendedBdigitalBbuildsBattachedB	abilitiesBwillisBwBtrailBstrongerBstiffBsavageBrootB	prisonersBplagueBoccurBnearbyBhudsonBheroicBgiftBcontraryBstrictlyBstoogesBlikingBlargerBlBgenerousBeditorB
conspiracyBclaireBcarlBboneBbareB	worthlessBworriedBupsB
terrifyingBsticksBsecretsBmustseeBmildBloadBjealousBiiiB
discussionBcaredBbrainsBanticsBachievementBwallaceBsoleB
overactingBlogicalBfreakBfittingBdorothyBdollB
disjointedBdiseaseBcouplesBwardBvalleyBsignsBruralBridiculouslyBpossibilityBplanningBnelsonBlistedBlabBianBhatredBgreyBetBdistanceBcriticBcoversBconnectBcitizenBchaosBburnedBbrideBbergmanBaffectedBtitanicBsadisticBpieBpaintingB
deliveringB	complaintBciaBcareersBargumentBandrewBagentsBoffendedBitllB	graduallyBdutyB	dinosaursBcookBcomicsBcomfortableBcamerasBbuttBbellBairplaneB
unpleasantB
universityBswearBscriptedBrussiaBrickBoverdoneBobscureBmenacingB	inspiringBdrawsB	dialoguesBdevoidBcupBconversationsBbleakBalltimeBwannabeBtwinBsidekickBofferingBnotablyBneighborBmeaninglessBirelandBintoBimplausibleBgrayBfurthermoreBexplicitB	everyonesBcraftBcombatBbroadBbluesBadamsBunwatchableB	travelingBstrikeBsentenceB	overratedB
overlookedBlyricsBlugosiBinvolvementBinvestigateBimprovedBfiredBexploreBcraigBbushBairportBwreckBvastBscreamsBscarlettBrandyB	primarilyBmoralityBmileBgoodnessBgenericBfingerBbathroomBwebBunbelievablyBstayingB
scientificBromanBrdBjuvenileBjazzB	eccentricBdrunkenBdrivelBchainBtranslationBthoseBsynopsisBspinBremovedB
prostituteBmummyBjudgingBhammerBdiscBdireBdifferencesB	dedicatedBcarolBblendBangelaBwinterBwarriorBtrioBspecificallyB	secretaryBlouBkiddingB	tarantinoBspikeBshiningBsecondlyBsadnessB	renditionBreachedBisolatedBinstallmentBhuhBheadedBfrankensteinBcoleBcanadaB	slightestBpunkBproceedingsBpornoB	executiveBerrorsB	disturbedB
depressionBbenefitBticketBteaBsubtletyBstagedBprisonerBpretendBidealBharderBeBdukeBdevotedBbulletsBbootBassassinByellowBtracksBstinkerBsloppyBsatanBrageBpoliticallyBordersB	newspaperB	murderousB	movementsBmarioBgentleBbuddiesBbatesBrecognitionBracialBpopcornBkitchenBholyBexaggeratedB	elaborateBcrackBconventionalBcomplainBbinBbakerBabsenceBviciousBupperB	terroristB
stereotypeBshipsBrobberyBrelativeBopinionsBmuseumBmixtureBmeasureBjustinB
influencedBfighterBfieldsBdignityBbeneathBwindsBwaxB
unbearableBrecordedB	psychoticBmotivesBmissesBliberalB	intriguedBintrigueBfurtherB
explainingBempireBdoomedB	decisionsBclumsyBcarterB	strangersBpreviewBpartlyBoddsBnedBmodelsB	lifestyleBhighestBhavingBfrancisBcruiseB	commentedBtracyBthrillBspellBshellBreunionBreliesBreducedB
philosophyBoccurredBneilB	manhattanBfatalBcardsBbrunoBtreesBtearBsportBshouldB	representBpushingB	producingBplannedBpantsBhopperB	gangstersB	explosionBenormousB	classicalBagingByaBtitledBthroatBthreateningB
simplisticB	scarecrowBrankBpoetryBoughtBonscreenBlightheartedBestateBdarkerBcoffeeB	buildingsBarrogantBagreedBaffordB	sacrificeB
recognizedBreachesB
populationBowenB	inventiveBillegalBevaBdamageB
convolutedBcaptivatingBbandsBamyBworkersBvegasBunintentionalBundoubtedlyBsuperficialBpaysBofficialBlucasBlipsBironB
homosexualBfrancoBfingersBfewBdistractingB	depressedBcampbellBblatantBwalkenBunfoldsB	similarlyBshineBruthlessBruinsBrangerBpassesBnarratorBmyersBmurrayBmenaceBmarthaBlimitsBkurtB
innovativeBexposedBduoBclothingBwingBversusBsolelyBprimaryB
introducesBhookBhonestyB
hitchcocksBfailingBexistedBdimensionalBdiamondBdepictsBalfredBtoddB	succeededBsplendidBselfishB	performerB	miserableBinvolveBgardenB	describesBcolonelBclosedBchildishBcatchesBblankBbetteB	yesterdayBstreepBphotographerBloadsBimaginedBgrabBcopiesBconB	comparingBaccompaniedBschemeB	provokingB	principalBnotionB
nightmaresBnicoleBmontageBmmBheartwarmingBgrandmotherBfondaBenemiesB	educationBdozensBcontractBclownBchoreographyBwetB
terroristsBtempleBteachingBtameBstylesBpatientsB	immenselyBhousesBhollowBfliesBeightiesBeatenBcrucialBcringeBconsistentlyBcircleBchannelsBbottleBborderBblockBamountsB	witnessedBwineB
unoriginalBtongueBthickBsmallerBroomsBritaBreportB
punishmentBparallelBnonstopBmistakenBmaintainBmafiaBinstantBfondBfocusingBexploredBdynamicBdustBdaBcountrysideBbelongBahBturnerBsoloBpurchaseBkickedBearsBconstructedB
consistentBcheatingBvividBtrainedBterryBstellarBsquadBslickBreedBprogressBplantBpennBmethodBloyalBkirkBhatesB	financialB	equipmentBemphasisBcoherentBcarreyBboldBadviseBwrappedBscoresB	satisfiedB
passionateB	mountainsBlayBjuniorBimpressBguestsBfooledBcurtisBcomposedBchestBcalmBblacksBadequateBurgeBsimilaritiesBrisingBpropsBlolBlegBjustifyB
helicopterBfrankieBformsBconvincinglyBbonusBagreesBadvanceBtrailersBtenderBstevensBpracticeBpotentiallyBoBfixBexoticBdiscussBconsiderableBclosetBblewBworeBwebsiteBtransformationBshirtBresponsibilityBpsychiatristBmargaretB	companionBbourneBblakeBannieBtonightBtieB
thoughtfulB	survivorsBsurvivedBruthB	recordingBpoeticBnorrisBmadonnaBleonardBgialloBenjoysBendingsBdownhillBderekBcoastBborrowedBalternativeBspringBrevolutionaryBresemblanceBpearlBlinkBknockBkeithBimproveBhomelessBhistoricallyBhilariouslyBfloridaB
disappointBdeliberatelyBdaviesBdatingBchickenBbetweenBuserBtriumphBtaughtBspiritsB	spielbergBsitsBrottenB	plausibleBonedimensionalB
mentioningB	masterfulB	franciscoBfolkBdivorceB	displayedBcusackBcarrieBbusterBbuffBbrandoBblowingBblondBalbumBtiesBtendsBsuitedBshelfBpuppetsB
pretendingBpaintedBnamelyBmateB
hollywoodsBelvisB
developingBdefeatB
cassavetesBbirdsBarrivedBamericasBabysmalBunevenBstaringBrubberBreflectBrapBoverwhelmingBhurtsBhideousBglassesBexchangeBearnedBdropsBdisappearedB	containedBclassesBbrutallyBarmedB	alongsideBaidBwritesBtadBsuitableB	suggestedBsmoothB	nostalgiaBmistressBlincolnBinaneBdocBbulletBadorableBwoundedB
underlyingBtwinsBswimmingB	seventiesBrivetingBresortBrabbitBportionBpartiesBparentBguessedBeditionBedgarB
creativityBcheatedBbridgesBborisBbollBbiteBadvertisingBabcByellingBusefulBunderstatedBtreatsBstiltedBshirleyBphilosophicalB	neighborsBmarchBjeffreyBhandedBegoBeasternBeaseBdirectsBbarneyBsolutionBslapBsafetyB
progressesBproceedsBmatchesB
journalistBjewsB
irrelevantBintimateBingredientsB	forbiddenBdrawingBcitizensBcarefulBwishedBwidowBswitchB	superheroBsquareBshakeB	relationsB
reflectionBquinnBpromisesB	laughablyBillnessBheartbreakingB	formulaicBfloatingBdollsBcoxBcontestBcombineBcliffBcinematographerB	charlotteBtwelveBtoplessBtankBphilBnervousBknightB	illogicalBendureBelegantB
corruptionBchasedB	capturingB	attitudesBapesBwoundBtrashyBthugsBsandlerBjesseB
horrifyingBgraspBfrustratingBfoughtBfasterBemperorBembarrassmentBdougBdestinyBdesiredBdentistBcannibalB	voiceoverB	vengeanceBtripeBsteelBsincereBsensesBseeksB
remarkablyBpopsBmotivationsBjoinedBjetBinappropriateBfrequentBdespairB
conditionsBcommandBcleverlyBcdBtimothyBstraightforwardBstinksBshowcaseB	pricelessBphraseBoccasionBmoodyB	madefortvBhuntersBhintsBflopBfacingB
enterpriseBdinosaurBdammeBcrystalBbannedBwealthBweakestB
unsettlingBunderstandableBtagBsappyB
nominationB	marketingBlustBlatelyBjawsBharveyBhamiltonB	guaranteeB
futuristicBfeastB	disappearBdancesBscopeBrolledBrebelBreactBpossibilitiesBmodeBmarionBjoshBincompetentBgloverBgloriousBgateBfrostBdancersB	conceivedB	communistBchallengingBbeliefsBbehaveBavoidedBwomensBunitBtylerBstarkBremainedBpremiereBpitifulBmarsBlegalBignorantBfrustrationBclosestBcircusBbuffsBbackgroundsBaliBwarmthB	wanderingBvirginiaBsplatterBsheenBschoolsBrockyB	relativesBpromisedB	prejudiceB	operationB	nostalgicBmonkeysBkarateBharoldBensuesBelectricBdesperationBcheBassaultB
apocalypseB
widescreenB	wellknownBtwilightBsettleBrelationBpagesBoutfitBmacyBgrudgeBdescentBcakeBcagneyBawakeBwishingBwackyB
vulnerableBvaluableB	survivingBsoftcoreBquotesBpeakBonlineB	mysteriesBlettersBhitchBdutchB	depictingBdemandBclaimedBbullBbaldwinB	affectionBwaitedBviewingsBteachersBspreadB	realizingB	purchasedB
popularityBnooneBkubrickBfaultsBeducationalBdefendBcgB	celebrityBbumblingB	behaviourBappreciationBalertBwatsonBteamsBspokeBspainBrunnerBrobotsBphonyBnycB	musiciansBmtvBmorrisBmayorBmankindBlandsBjoelBinvasionB	introduceBhartBgrahamBgenerationsBengageBeatsBeagerBclicheBcinemasBboxingBawhileBwarrenBunhappyBtransferBtouristBspeciesBspaceyBsoccerBrocketBpurposesBphantomBpassableBnutsBminimumBminiBinferiorB
equivalentBdrearyBdoomBdamnedBcrushB	climacticBchapterBarguablyBwavesBtomorrowBsagaBrepresentedBreadsBperformsBmightyBlowestBlitBlesBjuneB	greatnessBbutlerBbathB
artificialBarrivalBadaptationsBwifesBtromaB
suspiciousBstandoutBsevereBrefusedBpatienceBmiseryB
lonelinessBincomprehensibleBhughBgiftedBexperimentalBexceptionallyB
enthusiasmBelephantBdiscoveringB	corporateBcivilizationBchicksBboB	authorityBabsentBveniceBvaguelyB
underworldBturkishBteachesBsullivanBsoundingBrobinsonBresistBprizeBperryBpatriciaBpanicBnonsensicalBmoreoverBmerylBmachinesBloadedBlikewiseBlandingBgarfieldBdistinctBdefinedBdefenseBconstructionBcomposerB	commanderB	carradineBaustinBaffectBaffairsBwesBwellsB	secondaryB	resultingBpunBkingdomB	integrityBhungB	correctlyB	comediansBcolinBclaudeBbangBshowdownBscottishBroyalBrelyBpetersBnicholasBminsBmarshallBlastedBhitmanBfelixBfedB	expressedBellenBelBedgyBcomplainingBchainsawBcentersBcarmenBaprilB
activitiesBwivesBtroopsBtrafficB	testamentBstressBsmilingBsatisfyBreachingBpropertyBpressureBprepareB	nicholsonBmethodsBmassesB
literatureBhopelessBheadingBgusBguitarBfeverBdeeBcrossingBcheckedBbonesBbedroomBballsBastonishingBwildlyBunfairBtiresomeBtastesBsueBslaveB	senselessBraisesBquitBpursuitBpsychicBpreyBmasonBmaniacBkarloffBintentionallyBgreedBfirstlyB	excessiveB	conflictsBbronsonBawfullyBworkerBwisdomBwedBvisibleBtravestyBtoysBthruBthoughtprovokingBsimultaneouslyBsimpsonsBseverelyBsamuraiBrossBreceivesBquestionableBparanoiaBpackageBminBguardsBfestBcreamBcountsB	compelledBbernardB	alcoholicBalcoholBsixtiesBsandBpurpleBphotosBobjectsBmundaneBmaggieBexplorationBdylanBduckBdislikedB	dimensionBcannonBaustenB	upliftingBtribeBmuteBmetaphorBmanipulativeBlemmonB	imitationB	holocaustBhokeyB	grotesqueBderangedB
definitionBcubaBconservativeBconcernBcoincidenceBcatchyBcancerBbetrayalBaweBagendaBaaronBviceBvBunfoldB	trademarkBsunshineBstaleB	respectedB	resourcesBreleasesBmuppetB
landscapesBkickingBkennethBhumbleB	householdBglobalB	gentlemanBfemalesB
fascinatedBfamilysBexperimentsBcusterBconfessB
challengedBbayBwagonBvisitingBtriangleB
transitionBsungB	streisandBstrandedB	spaghettiBrudeBreelBpromBplightB
middleagedBharmBhackBfoulBdickensBcrueltyBcatchingBadvancedBaccessBwouldbeB	witnessesBvisitsBtrackingBsinkBsecretlyBscreensBreplaceBraymondBratB
passengersB	objectiveBmayhemBlivelyBleighBjudyBinvitedBimprovementBdressesB
disappearsBdianaBcostarBcomfortBcitiesBbranaghBblastB	biographyBaussieBassignedBantonioBwwBwornBwarriorsBvoyageBtroublesBtaxiB
simplicityBseldomBromeBretiredBrealmBphotoBoverlongBorderedBniroBnatalieBmartyBflairBexposureB	companiesBasylumBtownsBsurfingBstuartBrootsBrompBrobbinsBphillipsBnationsBmiracleBmentionsBmegB
lacklusterBjumpedBdonnaBdifferentlyBdiehardBdesiresB	crocodileBcrawfordBcorpsesB
complexityBclipBcampaignBanalysisBabusiveBabusedBsleazeBrichardsBrexB	repeatingBregionB	preciselyBnerdBmessedBlawsBgreedyBgarboBfalkB
exceptionsBernestBchoosesBB
underwaterBunawareBstunnedBstaticBspiritedB	slaughterBrequireBregardedBrecommendationBratesBpettyBmuddledBkyleBjordanBharmlessBgriffithBgabrielBfirmBfifthB	fashionedB	fantasiesBearB
complaintsB
compassionBchildsBcaryBbravoBberlinBaltmanB	alternateBuweBunseenBsignedBsalmanBridesBreferredB	occasionsBjewelBinterestinglyB	inabilityBgemsBdevilsBdemiseBcubeB	completedBchoreographedBargentoBwidmarkBwidelyBvirtualBunrealBtrendBsneakBshakespearesBrestoredBremarksBrandomlyBpalB
misleadingBlengthyBkittyBjoinsBinsanityBhelloBfemmeBdoubtsB
directionsBdebateBcreditedBcreatorBcluelessB
christiansBchessBbullockBbrooklynBappropriatelyB	addictionBaccuracyBunpredictableBsymbolicBshoutingBsamuelBrupertBroofBproBnailBmollyBleoBinventedB
inevitablyB	immediateBhughesB
expositionB
carpentersBbewareB
accuratelyB–BwintersBwheresBstumbledBsparkBpolishBpigBownersBnoisesBmasterpiecesBliftBinvestigatingBexorcistB	encourageBconfrontationBconceptsB	ambiguousBaimBaidsBtackyBsubparBstatueBshoulderBresembleB
positivelyBmontyBliteraryBhustonBfoxxBearlBdesignsB
controlledBcapitalBbeanBbargainBwingsBvictoryBreevesBrealisedBpotBplanesBoverlookBodysseyB
obligatoryBmusicianB	murderingBmoronicBmaidBlockBliftedBireneB	interestsB	ignoranceBhippieBharrisonBgolfBgodzillaBfortyBforcingBfisherBfeedBdysfunctionalB
destroyingBcrispB
containingBcastsBadoptedBwizardBtopnotchB	tastelessBsunnyBsufficeB	subjectedB	strongestBsingersB	sillinessBsarandonBquietlyBnewmanBnemesisBlosersBiconBfillerBfavourBeugeneBdistributionB
describingBdenzelBdameBcloseupBchampionBbusinessmanBboomBbabiesBbabeBassumedBacidBvoightBvoicedB	unlikableBunexpectedlyBundeadBtopsBtigerBsometimeBsignificanceBrouteBritterB	primitiveB
presentingB
portrayalsB	policemanBmapBknockedBjulietBiraqBinteractionsBinjuredBhindiB	generatedBdemonstratesBdaddyBcasualBamusedBaddressBvinceBtapBsharesBrepresentationBremoveBquaidBpostedBpolishedBpanBoutingBoriginBmuppetsBmeritsBmacbethBlasBlanceBkathyBjoeyBjennyBjackassBgregBentitledBegyptianBdivineBdisneysBdelicateB
confidenceB	computersBcarlosBbudBbiasedBassassinationBwillieBwhollyBsurvivorBshoddyBsendingBrenderedBrecycledB	prominentB	profanityBnewlyBnetBminusBinexplicablyBhungryBgrainyB
frightenedBfascinationBexploresBdroveBdrakeBdomesticB	christinaB	barrymoreBbanalBatlantisBanyonesB
additionalBactiveBwardrobeBvelvetBveinB
unlikeableBthompsonBswimB	supportedBstillerB	speciallyBreferBpaleBownedBnodBmBlocalsB	heartfeltBgereBfiB	evolutionBdrinksBcopeBcoBchoppyBbettieBbelaBventureBtransformedBsuspendBsomedayBsimmonsBsharedB	rochesterBremakesBremadeBpostersBpitBorangeBministerBloyaltyB
inaccurateB
hopelesslyBfoolsBempathyBdarnBcueBconanBbunnyBauthoritiesBapeBwooBsumsBrodBregardsBraisingBpreachyBparadiseBopportunitiesBmilkBlocatedBkissingBjulianBinspireBincidentallyBhatsB	hackneyedBgrabsBdevicesBdementedBdeafBcrashesBchuckleBchorusBbeverlyBassumingBwongB	threatensBsurgeryBstandupBsoonerBrejectedBreaderBpeteB	nightclubB
insightfulBinconsistentBfeelgoodBdirtBdaltonB
conscienceB
challengesB	brutalityBbarsBballetBangstBambitionB	acclaimedBweaknessB
underneathBsliceB
respectiveBrelaxBpreviewsBnbcBmoralsBmodestBmichaelsBmarineBlifelessBlegacyBlaurenBlatinBitemsBiqBherosBhearsBgloriaBfreaksBfillingBfiftyB	exploringBdreckBdestroysBdesertedBcrittersBcouchBauthorsBattorneyBandreByardBwitchesB
suspensionBstudyingBstoresBspockBsnipesBshortcomingsBservantBronaldBriotBrespectableBresidentBrecordsB
psychologyBpredecessorBpoemBoliviaBminorityBmailBlinersBhestonBgoodbyeBerikaBenBdifficultiesB
derivativeBdanielsBdanesBchargedBattendBabruptBwretchedBvisionsBstringsBstonesB	shouldersBservingB	septemberBrefuseBqueensBpreposterousB	preferredBplayboyBownsBoutfitsBnorthernBmythBmildredBlimitBkramerBkayBinvitesBheavensBfirmlyBcontinuallyBconsiderablyBcommunicateBalecBwakesBunderstandsBthreadBshorterBscreenwritersBrisesBreflectsBorsonBoptionBmillBlastingBkapoorBhilarityBheightsBflashesBfiresBevansBeternalBelsesBchristieBchampionshipB
accomplishB
acceptanceBwweBvulgarBsnlBshakyBpoisonBnativesBmutantBmatchedBjacketBivBinsipidB
imaginableBhostageBheroinBgroundbreakingB	demandingBdeceasedBdatesBcormanB
convictionBconradBbradyBbiblicalBbesideB
basketballB
adolescentB
wildernessBwigB
weaknessesBvitalBtechnicolorBsublimeBsnowmanBskeletonB	selectionBseBscrewedBropeBresolvedBreadersBracesBpuzzleB	programmeBponyoBpilotsBorleansBnolanBletdownBkeenB	identicalBhomerBgypsyBfiancéB	disgustedBdeletedBaspiringB	amusementBwellwrittenBweatherBunexplainedBsupplyBshawBracingBpokerB
montgomeryBmarkedBjudgmentBjadedBinsistsBinfectedBhootBfiftiesBemergesBdemonicBcrazedBconnectionsB
caricatureB
terminatorBsurroundingsBsubtlyBstagesBshesBsellersBsciB	reluctantBrefersBratsBpartnersB	partiallyB
motorcycleBlilyBlaborBjealousyBhealthyBgriefBfuriousBfistBethnicBenteredB
discussingBdesignerBdanishBchillsBcareyBbtwBbountyBbmoviesBbluntBagencyB	afterwardByepBwrapBstareBsmilesB	sincerelyBshakingBseymourBrubyBredeemBpushesBprovocativeBparadeB	paintingsBmercyBmatesBliBkungfuBhistoricBheartedB
guaranteedBfrozenBeditBdroppingB
detectivesBdealerBdamonBcreepBconfrontBbombsBblobBaddictByoungestBvotedBvisitedBvibrantBtunnelBtopicsBthrilledBtBsuburbanBstoppingBsteadyBsimpsonBshtBsharingBsandyBrustyBrourkeBrippingBrespectivelyBrapidlyBpennyBpalaceBmindedB	mentalityBmasksBmarvelBmarriesBmachoB	imaginaryBfunctionBframedB
farfetchedB	exquisiteBelevatorBdisguiseBdependsBcornBconventionsB	attackingB
astoundingBwalshB	uniformlyB	terrifiedBtemptedBtailBstudiesB	stretchedBskitsBsitcomsBshowtimeBsaleBrooneyB
professionBpredictBnailsBmonkBlunchBleesBlabelBinternalBfayBerrolB
engrossingBearnBdressingBdoseB
difficultyB	dependingBdallasBcoupledB
continuingBclaustrophobicBcenteredBcemeteryBcaliberBbullyBastaireB	admirableBwaltBvastlyBvainB	unrelatedBtravoltaBthievesBsyndromeB	strengthsBstickingBrangersBquentinB
psychopathBjillBjarringBhugelyBhoBhelplessBhelpfulBhankBhamBgoldbergBglobeBgenderBfeatBexpectsB	evidentlyBemployedBdelBconveysBbiopicBbilledBbeholdBbamB	absurdityBwindowsBtraitsBtowersBsurvivesB	separatedBscotlandBschlockBrussiansB
richardsonB	residentsBramboBprequelBolBnyBmorallyBmontanaB
kidnappingB	ingeniousBindependenceBhulkBhandlingBhammyB	gentlemenBgatherBfuryBfrancesBfilthBeventualBelviraBdisgraceBdepthsBdebbieBcountyBcheekBbuffaloBbrosBbrendaBattractBakinByawnBvoodooBusersBtrampBstumblesBspeechesBshoreBroommateBrecognizableBranchBqBpromoteBpacificBolivierBnieceB	misguidedBlushBlayersBhopkinsBhaplessBglowingBgirlfriendsBgalBenteringBeliteBdudBclaimingBcanceledB	blatantlyB	befriendsBwellmadeBvolumeBveraBuncutB
translatedBrhythmBpoliticiansBpiperBoffbeatBmoronBmitchumBmessyBmeltingBmalesBlifesBlastsBjacksonsBinfoBimmatureBhydeBhayesBgingerBforgivenBethanBdisgustBculturesBcravenBchristianityBbugBbrosnanB	breakdownBbootsBbonnieBalteredBakshayBaddictedBaccountsBabruptlyBtaBsydneyBshootoutBshadesBriversBrerunsB	referringB	receivingBprovingBprayBpleasingBmixingBmannBlenaBleadersB
indicationB	incidentsBheightB	deservingBdanaBcostarsBcontributionBcolumbiaBclichesBbutcherBbogartBbelushiBactivityBwaitressBughBtorontoBtacticsB
suggestionBslimyBskullBpunchesB	paramountBmuslimBmotiveB
mechanicalBloyBlethalBintentionalBinstinctBhomosexualityBhomicideBhidesBhangsBhandlesBgregoryBgangsBgamblingBflashyBfiringBeternityB	energeticBdevastatingBdefeatedBdeclineBcompeteBcomparisonsBcelebrationBcaricaturesBburstBbentBarthouseBansweredBalleyBafghanistanByoureBwesleyBtapesBtapedBspencerB	spectacleB	sentimentBscrewB	satiricalB
restrainedBraveBoutrightBmealBhoganBgeorgesBfogBexaminationBescapingBemailBdorisBcloneBbostonBartsyBaboardByoutubeB	startlingBsnuffBsiblingsBshoppingBscoobyBsaraBsaintBrifleBreportsB
phenomenonBleatherB	justifiedBglanceBgalaxyBformedBfoolishB	establishBelevenBcushingBcannesBbuseyBautomaticallyBassureBanticipationBwoundsBvotesBtrainsBswingB	suspicionBsuitablyBstairsBsopranosBsketchBsidBsherlockBsamanthaB	publicityBprogramsBpmBpiratesBpenguinBoldsBnicolasBmallBlegendsBisraelBinhabitantsBhomesBgandhiBgainedBfluffBerrorBdurationBcostelloBcommitsB
commentingBbaconB	backwardsBapproachingBacceptsBwandersBwanderB	victorianB
undercoverBtubeBsymbolB
stunninglyBsmellBsensibleBseedBscratchBrootingBreignBnervesBmesmerizingBlionelBkansasBjulyBinstitutionB	fortunateBfeministBfactorsBeyreBdunneB
disastrousBdillonB
despicableBdawsonB	convincesB	confidentBcleaningB	camcorderBbudgetsBbillsBbikeBbeatlesB
approachesBanytimeB£BtriviaBtrialsB
threatenedBslipBsergeantB	sarcasticBromeroB	repulsiveBpfeifferBpcBpaxtonBpatternBmorbidBmiyazakiBlimitationsBinformativeB
industrialBhainesBhabitBgoodsBgluedBfordsBexteriorB	explodingBenhancedBdubB	convictedBconvenientlyBcombinesBcollectBcoatBboastsBbitchBbarrelBauthenticityBwhaleBwelldoneBwarmingBunsuspectingBtossedB	tolerableBsupremeBpierceB
phenomenalBpeculiarBpayoffBnormBmeantimeBlouiseBlinkedBjuryBiconicBgrassBframesBexposeB
expeditionBentiretyBemergeBedieBdestinationBcycleBcookingBconsiderationB	christineBchoosingBbacksBavidBarrivingBwagnerBvocalBvileBtoolBstardomBsailorBrewardBrescuedBrandolphBprestonB
possessionBparanoidBoverbearingBoutlineBoutdatedB	orchestraBnephewBmodestyBlongestBleonBkidmanBkarlBjudgesBitemBinformedBimmortalBhippiesBhayworthBgerardBfishingB	fastpacedBenthusiasticBdubiousBdowneyBdisabledBdilemmaBdiazBdelightfullyBcoloursBcheerBalmightyBadoreBwendyBwardenBtokenBthrustBstudiedBstrokeBsteerBspreeBsinkingBseemingBscariestBritualB	principleBpainterBoldestBmoronsB
mannerismsBlowkeyBlastlyBingridBfataleBexpenseB	endlesslyBdiverseB	disguisedBdestinedB	departureBdeniroB
deliberateB	decidedlyBcrackingBcopiedB
convenientB
colleaguesBcliveBcharacterizationsBcentsBcattleBbwBboxerBbeltBbeattyBasiaBantsBamazonBaceBwiselyBwashedBupsideBunimaginativeBtremendouslyBtraceBtendencyBsorelyBrussoB	rewardingBrespectsB
rebelliousBpretendsBpornographyBpirateBphysicsB
perceptionB	ourselvesB
moviegoersBmedicineB	intricateBimpliedB	horrifiedBhiresBhiltonBhavocBhauntBgoalsBfundingBdreadBdividedBdismalB
contributeB
confrontedBconcentrateBcentreBcaseyBaxeBattendedBassuredBarkB
approachedBwireBwhitesBunsatisfyingBsweptBsoupBsosoBseniorB	screwballBromeoB	regularlyB	publishedBpassageB
misfortuneBmedievalBlivBlikeableBjulesBinteractBinexplicableBhkBgripBgratefulBdinerBdepictBdenyBdemonstrateB	deliciousBcurlyBcrashingBcontroversyBconcludeB
commitmentBbroodingB	brainlessBbelowBavoidingBartworkBarrestBarcBamitabhBaffectsBwilderBwartimeBtipBtcmB
sympathizeBstorysB	spidermanBsparksBrothBreminderBpsycheBposingB
paranormalBnunBnopeBnominationsBnaughtyBmouthsBloisBlendBlaurenceBjerseyBjanetBintroBinteriorB	insuranceBinherentBincestBidolBhornyBhighwayBgearBfadeBexpectationBdetractBcriesBcoreyBclooneyBchoppedBcarellB	breakfastBbreadBashleyB	acceptingB	wellactedBveteransBvehiclesB	underwearBtravisB	translateBtowerBtoniBstumbleBstalkingBstabbedBshiftBshatnerBsessionBsellsBselectedBsaltBrhettBrealizationBpursuedB	preparingBpredictablyBpostwarBpoleBpaulieB	parallelsB	overblownBominousBnutBnetflixBladderBjessB	incapableB	housewifeBheavyhandedBgrowthBfreakingB
foundationBflowerBeducatedBdumpB	criticizeBcowB	cigaretteBcharmsB	candidateBboyleBbowBbitingBauditionB
assignmentBarabBanyhowBaircraftBtheatresBsteamBspiralBsfBsafelyBrollsB	remembersBrainyBquestioningBpraisedBneatlyBmormonBmillionaireBhandheldBgrinchBgodawfulBgilbertBghouliesBgatesBfollowupBfarmerB	extensiveB	explosiveBexitBdumberBdriveinBdivorcedBdiscoB	delightedBdadsB
complimentBclockBcharacteristicsBbulkBbowlBapplyBadditionallyByouthfulBwtfBwiderBswitchedBswearingBsixthBreliableBrebelsBradicalBpursueBpromotedBpoundsBpossessBowesB	organizedBmoeBmirrorsBmarlonBlongtimeBlongingBlicenseBkurtzBklineBinsightsB	incorrectB
improbableBexploitsBeverettBdrabBdevitoBdeerBdamagedB
conventionBcontributedBcloudsBcasinoBbuysBbreedBbikerB	armstrongBappliedBappealsB
advertisedBabrahamBwastesB	valentineBtonBtomatoesBsubsequentlyB	sickeningBseedyBsandersBreviewedB
resemblingB	relevanceB
relentlessBrearB	realitiesBrealisticallyBquarterB	practicalBplantsBoldfashionedBneuroticBmosesBmonroeBmobileBlolaBleapBjulietteB	inventionBinspirationalBindianaBimmenseBhookerBgoodlookingBgableBfullerBflowersBfillsBentriesBensureBelmBegyptBderBdeputyBczechBcounterBcoffinB
chroniclesBcbsBbittenBbalancedB	associateB	architectBwaynesBvomitBvintageBunsureB
undeniablyBultraB	suspectedBsuckerBstylizedB
statementsBspanBsourcesBslavesBscoredB	scenariosBsatisfactionBrepliesBposesBpoppingBpathosBoriginsBobservationB
noticeableBmiloBmiikeB
mediocrityBkidnapBjohnsBinterviewedBherzogB	harrowingBgimmickBflagBfilthyBduvallBdramaticallyBdjBdistractionBcyborgB
comprehendBbondsBbabesBuniformBuhBturdB	submarineBstationsBshiftsBshanghaiBsaneB	repressedBrampageBplottingBmumBmensBliteralB
legitimateB	isolationBghettoBfulcisBfrontierBfloodBexperiencingBexcruciatingBerBembarrassinglyBdustinBdeterminationBcritiqueBcrippledBconveyedB
collectiveBclayBclassyBbarkerBaimingB
accessibleBzBwrightBwhoreBwhomBwheelB	vignettesBticketsBswallowBsubwayB	spaceshipBsleepsBsharonBrememberingBragingBpoundBphotographsBphillipBottoBorganizationBollieBobtainBnewcomerBmutualBmarcBmamaBkumarBisabelleB	intenselyBhallmarkBgrabbedBgorillaBgiantsBgesturesBgapsBgadgetB	finishingBfarrellBencounteredBembraceBdumpedBdragonsBdominoB
definitiveBdazzlingBcrawlBcommunicationBcollinsBboogieBbelievabilityBbeggingBatomicBapplaudBamidstBadaptionBweakerBunderBtokyoB	telephoneBsweepingBsubtextBstuffedBstrictBskilledBshepherdBseasonedB	reviewingBreservedBrecoverBrainbowB	monologueBmasterfullyBistanbulBinsultsBinmatesB	inclusionB	immigrantBgroundsBgarageBfiancéeBexploitBexcessBestablishingBehBdownsBdownfallBdolphBdictatorBconfuseBclashB
cartoonishB	argumentsB	aftermathBafricanamericanB	absorbingBabortionB
werewolvesBvanessaBtripleB	tormentedBtitularBswayzeBsugarBstellaBstargateBslappedB	possessesBpaintsBpadB	offscreenB
noteworthyBmidstBmaximumBkoreaBjudgedBjoannaBjawBinsistBhybridBherculesBheelsBharborBgravesBglenBgarnerBfreakyBflamesBfeedingBfactualBdocumentBdisorderBcorporationBcobraBchaplinsBbittersweetBaugustBarguingB	aestheticBwhiningB	unnaturalB
unfamiliarBunderstatementBtoothBtongueincheekBstalloneBspiderBslimBselfindulgentB	seductiveBseatsBsanityBsangBroundsBromancesBrobbersBrightlyBrewardedBqualifyBpromptlyB
principalsB
pedestrianBpeckBpauseBpalanceB	originalsBnoveltyBnolteBnewerBmonkeesBmitchBmirandaBlynnBluisBlucilleBlighterBliarB	glamorousBgiganticBflimsyBexplodeBevanBemployeeBelectionB
distractedBdefineBdecidingBdarrenBcontemptBconsBcomplicationsB	cameramanBbenefitsBbellyB	backstoryB
aggressiveBachievesB	viewpointBverdictBustinovBunderdogBtvsB	toleranceBtabooBshocksBshelleyB	shamelessBseveredB	sentencesBsalesBrugbyBresumeB	redundantB
protectionB
profoundlyBpizzaBpeacefulBpamelaBotooleBothelloB	obsessiveBnoticesB
meanderingBlocateBlionsB	lightningBindifferentBhepburnBhartleyBglimpsesBfuelBfrogBfluidBflavorBexpertsBexistingBdefiesBcreekBcompositionBclimbBclanBchuckyBcheaplyBcerebralBcabBbratBblamedBbanterBartyBagonyBaffleckBwangBvanityBunsympatheticBuniformsBtheoriesBsubstantialBslashersBsergioBsentimentalityBroutinesBpointingBpamBmurkyBmiamiBmenciaBmelodyBmccoyBloweBlesterBlendsBkeysBkermitBjoBjanBintroducingBglaringBfrightB	firstrateBexcusesBexclusivelyB	estrangedBeggBdistractBdiamondsBdemonstratedBdealersBcoveringBcheersBcellsBcaperBbuzzBbsBbasingerBacknowledgeBwhinyBvoidBvalidB	unusuallyBunattractiveBtargetsBspadeBsoughtBslyBskitBshoeBrejectsBrehashBprosBproductsBpotterBpierreBpansBnathanBnarrowBmaloneBmacabreBkentBinterruptedBintactBidealsBheatherBheadacheB	graveyardBgibsonBfragileB	exploitedBdumbestBdooBclubsBclichedBchaoticBcampusBboothBbetrayedBbegB	assembledB	affectingBadmiredBaboundB
witnessingB	variationBvalBunderdevelopedBunconventionalBspineBslideBskinnyBsissyB	reflectedBreaganBrationalBrapistBramblingBpenelopeBoceansBnovakB	neglectedBmerchantBmarinesBlandedBinvestigatorBignoringBgromitBgeekBgarlandBfranticBfortBforemostBflowsB
executivesB	employeesBearnestBdistressBdebtBcomaB	colleagueBclausBcassidyBcasperBbillingBbennettBbelleBbaddiesBaztecBassumesBartistryBalaBadrianB	admissionB	addressedBacquiredBabbottBxfilesB
wonderlandBwashBvivianBverbalBtrivialBtraumaBswingingB
suggestingBsportingBspitBsettledBscroogeBrivalsBreverseBresultedBreidBredneckB
politicianBploddingB	patrioticBparksBpacksBnetworksBnerveBmysteriouslyBmisunderstoodBmissileBmelissaBlelandBlavishBjoiningBisraeliB	instancesBinjuryBinfluentialBhansBgoldieBgenerateBexcellentlyBevokesBebertB	disregardB	discussedBdaytimeBcrookedBcoppolaBconsistBcomebackBclerkBcirclesBcharliesBcharacterisationBbattlingBalisonBwarrantBvariedBupdatedBupdateBupcomingBunBthreadsBtherebyB	stephanieBspiesBsensationalB	salvationBroryB	remindingBpsychedelicBpokemonBpocketBphaseBparodiesBpaltrowB	mythologyBmudBmockBmickBmessingBmadsenBlizBlipBkellysB
immigrantsBgundamB
graduationBfoilB	festivalsBfenceBexcruciatinglyBelliottBeagerlyBduBdriversB	dominatedBdebraBdashingBcrossedBcouldntBconsequentlyBconsciousnessB
complainedBclunkyBclarkeBchopB	breathingBborrowBalvinBadsB
admirationBvividlyB
villainousBuncannyBstreamBsterlingBseduceBroundedBroughlyBriderBriceBricciBrebeccaBpythonBportmanBpolanskiBpathsB	overboardB
optimisticBoharaBobserveBnormaBlynchsBlucioBjudeBindicateBhystericallyBfruitBfodderBfarleyBfadedBengineBedwardsBebayB
depictionsBdcBdaylightBcookieBcontestantsB
comparableB
celebratedBcapacityB	ambiguityBalterBaidedBwarnsBtonesBsustainBstereotypedBstartersBstalkerBsourBsmashB
shockinglyBservicesB
resistanceBrepublicBporterBparkingBpapersBmoviemakingBmonicaBmeteorB	macarthurB	knightleyBkinnearBjurassicBjuanBhorizonBhitlersBhenchmenBgloomyBfritzBexplodesBentranceB	enigmaticBelderBeggsBdoveBdiBdeliciouslyBdaffyBcrownB
confessionBcohenBcarsonB
captivatedBcampingBbustBbridgetBbishopBbikiniBbanksBbacallB
antagonistBabstractBwhereverB
wheelchairBupbeatBumBtombBsubgenreBsubduedB	stretchesBsociallyBsleeperB	schneiderBrobocopBreplacementBproducesB	preachingBpainsBourBomenBoctoberBmotionsBmomentumBmannersBlonesomeBloBknocksBiranB
highschoolBherbertBgoatBginaB	followersBflowingBfiennesBdiaryB	depardieuB
denouementBcursedBcourtesyBcontrolsB	considersB	consciousBcapeBbombingBbaronBarrayBaptBantiheroBaliciaBadaptB
youngstersBvergeB
unansweredBturmoilBtireBsweatBsummedB
stepmotherBstackBsophieBsketchesBservantsBsensibilityBscreenedBscorseseBromanianB	perceivedBpattyBparticipateB
overweightBordealBnuancesBnuancedBmotelBmixesB
melancholyB	maintainsBlurkingBlupinoBlibertyBinaccuraciesBhowlingBhiBfuzzyB
forgettingBfeminineBexpertlyB
excellenceBepicsBedisonBeconomicBeagleBdodgyBdegreesBdeadpanB
committingBclimbingBcarnivalBcarlaBcampsBbuttonsBbordersB	attendingBwhoopiBvcrBvaughnBtoxicBtaraBsymbolsBswedenB	surrenderBsurfB
substituteBstreakBsmarterBsleptBrosesBrollerBrespondBresolveB	releasingBrazorBprofessionalsBpolicyBplateB	officialsBnutshellBninaBnarratedBmysticalB	murderersBmichealBmaskedBlunaticB	libertiesBleanBlansburyBjacksBinclinedB
incestuousBhuntedBhostileBhintedBgutBgoodmanBghastlyB	furnitureBfulfillBfeebleBenhanceBenduringBdrillBdixonBdismissBdesBcloudBclaraBchillB	centuriesBcanyonBbreastBboobsBallensBwovenBunclearBswitchesBsundanceBstonedBspellingBsmugBsleepyBsinksB	sensationB	sebastianBscottsBscarierBsarcasmBrousingBripsBreluctantlyBreadilyBrajBproceedBpreciseB	populatedBpimpBpbsBopenlyBoffendBmadisonBlooneyBloganB
lieutenantBjoseBivanBinsertedBincreaseBillusionBgrislyBgraysonBgravityBgeorgiaB	gatheringBgameraBframingBforgetsBfontaineB	eastwoodsBdynamiteBdynamicsBdistinctiveBdistinctionBdetroitBdespiseB	customersBcrooksBbiasBbegsBbarbraBaustensBarquetteBwouldntBwinnersB	warehouseBvanceBunderstandablyBtightlyBtaxBstatingBstableBsolvedB	sincerityBshotgunBsheepBshawnBsalesmanBrickyB	retellingB	renderingBrenaissanceBregimeBprostitutionBprostitutesB
progressedBpoetB	pleasuresBplanetsBphoenixB	obscurityBnoahBmarxBlopezBlaunchBitdBfrontalBfacilityBexwifeB
enchantingBelectedBdiveBcousinsBconfinedBcokeBclickBcheeringBchatBbennyBbendBbakshiBadmitsBwillardBwayansBwakingB	verhoevenB
timberlakeBthirtiesBtautB	stupidestBsternBstagingBstabBspinalBsgtBserialsBsegalB
scratchingBsammyBsabrinaBrunawayBreneeBrelentlesslyBpsychologistBprogressionBpoppedBpivotalBperverseBoverusedBnivenBmyrnaBmustveBmanipulationB	lookalikeB
liveactionBkrisB	injusticeBhornBhollandBhesitateBforgivenessBfetchedBengineerBdiggingBdeliveranceB
criticizedBconsequenceB
compensateBceilingBcarnageBbooneBbegunBbearingBbattlefieldBbashingB	awarenessBaudreyBarticleBanxiousBanticipatedBallegedB	alexandraBthirstBsurroundB	subtitledBstewartsBstalkedBstakeBspringerBspiceBshuttleBsensitivityB	scriptingBsatanicBrodneyB	robertsonBrobbedBpickupBphiladelphiaBpeersB
officiallyBobservationsBniftyB
nauseatingBmovieiBmermaidBmarilynB	malkovichBlundgrenBlenoBlambsBkinskiBinviteBinvestedBinsomniaBinconsistenciesBimoBhurryBheapBgapBdukesBdrawingsBdistributedB	curiouslyBcrossesBcreepsBcracksBcountingBcobbBcladBchargesBcelebritiesBbuildupBbrettBbothersBboardingBblockbustersBbatsB	antonioniBamirBallyBzanyBzaneBwrestlerB	wholesomeBweaverB	unfoldingBtransportedBtransferredBtolerateBtherapyBtechBsunsetBstefanBspoofsBsinsBsightsBshannonBsensualBsenBsassyB	revolvingBrelaxedBrelatingBrecreateBpompousBpoeBperiodsBpaulaBpalmBoutsetBoffenseBobrienBnortonBmelvilleB	magicallyBlordsB	languagesB
laboratoryBitaliansBhypedBholmBhannahBgraduateBglendaBgeinBfortiesBflewBfarmersBfailuresBeconomyBduelBcrosbyB
courageousBcoupBcocaineBclarityBchaptersBchaneyBbondageBbloomBbachelorB	anthologyBalliesBadmittedBaccompanyingBabroadByellsB
witchcraftBvibeBumaBtunedBthugBthoBteddyBtargetedBsurgeonB
sufficientBspottedBsighBscheduleB
scarecrowsBrevoltBprologueBprogrammingB
principlesBplatoonBpinBpeggyBpalsBnearestB
monotonousB
monologuesBmalcolmBlangeBkubricksBjewBjacquesBiraBintendB	instinctsBinsertBinchBhypnoticBhoneyBhatefulBgrierBfundamentalBfranklinBfilmographyBfileBfierceB
favouritesBfanaticB	enchantedB	emphasizeBdraggingBdevotionBderivedB	courtroomBcoburnB	civiliansBcaptiveBborrowsBbeardBatlanticB
astronautsBassociationBampleBabominationBByellB	vigilanteB	underusedBtroopersB
traditionsBthunderBswampBsuperfluousBstrungBsnappyB	skepticalBshempBsethBscriptwriterBrookieBrecklessBrainesBpuzzledBpursuingBpubBproportionsBpropBposeB
portugueseBphoebeBpenaltyBpairingB
ostensiblyBoccupiedB	obliviousBnightmarishBlumetBltBjodieBinadvertentlyB
improvisedB	impendingB
identifiedBhenchmanBguinnessBguineaBgrandmaBfreezeBflipBfairnessBcowardBconvictBcheerfulB
casablancaBcarolineBbuffyBbrendanB	boxofficeBblessBbachBauteurB	assassinsB	ambitionsB
accidentalBweeklyB
transformsBstrainB
similarityBsidewalkBsicknessBshelterBseinfeldBromaniaBrequestBrepeatsB	recurringBpriestsB	premingerBpollyBparticipantsBpalmaBomarB	obstaclesBnestBlistenedBlimpBlensBknightsBkilmerBjolieBinterpretationsBinsultedBinformsBignoresBhisherBhermanBheathBhawksBhawkeBhardenedBhalfhourBguardianBfleeBflashingBfidoBelectricityBeffortlesslyBdeniedB
dedicationBcynicismBcrushedB
criticismsB	confrontsBcombsBchristyBcheatBchamberBbubbleBbleedBblackandwhiteBbearableBarrangedB
armageddonBappliesBapplauseB	announcedB	allegedlyBachievementsBzoeyB	worldwideB	transportBtierneyBtenantBstrippedBspoilingBsmoothlyBsignificantlyBsexistBsenatorBrogueB
retrospectBreportedBrantBramonesBpuppyBprotBprofileBpornographicBplayfulB	pervertedBpaBovershadowedBninetyBnielsenBnerdsBneedingBnannyBmythicalBmooresBlayingBlargestBlambertBknockingBkeanuB	katherineBkahnB	judgementB
increasingB
idealisticBhopBhawnBhardestBgreatsBgrangerBgovindaBgalleryBfraudBflippingBfleshedBevokeB	espionageBenvyBedgesBearliestBdrumBdominateBdodgeBdistinguishedB	coworkersBcounterpartBcontrollingB	confirmedB	combiningBcheadleB	brazilianBboostBbogusBbloodthirstyB	awakeningB	astronautBasterixBarrowBantiBangelinaBadvancesB	abundanceB
winchesterBwhoveBviscontiB	villagersBunconsciousBunappealingB	traumaticB	terrorismBsylviaBsuckingBstrainedBsteeleB	spotlightBslowerBshapedBsectionsB
sacrificesBrevelationsBrejectBredfordBrapidBraidB	qualifiesBprotestBphonesBpatriotBongoingBneroBnerdyBlangB	irritatedB
influencesB	homicidalB
happeningsBhackmanBgeraldBgatheredBfixedBexplanationsB	europeansBensueBelmerB
eisensteinBdudleyBdoyleB
displayingBdeedBdaisyBcustodyBcummingsBcostnerBcontributesBconclusionsBcoloredB	collectorBcmonBcircaBcharacteristicBcasuallyBcandleBbutchBbryanBbreatheB
boundariesBbitchyBbartBbaitB	awfulnessBautoBantonBanchorBwormsBweeBwaltersBvegaBvapidBvaderBunnecessarilyBuneasyBtransitionsBtimonBsuicidalBsmithsBshockerBseriousnessB	semblanceBscarfaceBruiningBrioBremarkB	remainderBrealisesBraysBprofitBperspectivesB	overtonesBmomsBmobstersBmissionsBmclaglenBmarathonBmanipulatedBlorettaBliuBlabeledBkurosawaBkristoffersonBkneesBkindlyBislandsBinhabitB
infinitelyB
incidentalB
impeccableBhockeyBhahaBgrooveBgoodingBgiBfunkyBfriedBfirthBexploitativeBevolvedB	dreamlikeB
distinctlyBdetachedBdeskBdecencyBcunningBcrowdedBcrashedBconsumedB
commandingBcharityBcavalryBburtonsBbossesBbonBblokeBbasilBavoidsB
assistanceB	arroganceBzombiBwonBuptightBtrapsBtheftBtalkieBsystemsB
surrealismBsteamingBstadiumBshylockBsheetsBsecureBschemingB	scatteredB	satelliteBrudyB	restraintBrailroadBportionsBpoirotBpaddingBoveractsB
outlandishBoutcastBninjasB	necessityBmortalBmindsetBmillsBmandyBlorreBlindsayBlilBlightweightBlestatBlaysBkidnapsBkathrynBjointBinformBimpliesBhometownBhmmBhawkBharvestBhandicappedBfratBfinneyBfetishBfartBfadesBexcelsBevelynBestablishmentB
engagementBemmyBelijahBedmundBeaterBdwarfBdreamingBdistrictBdirecttovideoBdinBdandyBcounterpartsBcollapseBcliffhangerBchevyBcapBblinkBblazingB	backwoodsBawryBawaitingBarentBappleBantwoneB	anonymousBandreasBabandonB	zellwegerBwipedBwhereinBunstableBtobyBtestingBtellyBtablesBssBsentinelBscarredBrougeB
retirementBrespondsBrelyingBpredecessorsBprecodeBpaycheckBoverwroughtBmobsterBmiraculouslyBmeredithBmarryingBmarketedBmarcusBmadmanBlawyersBhopefulB
hellraiserBgillianBgielgudBgeeBfederalBfaintBernieBdustyBdowntownB	diversityB	determineBdenisBdeemedBchowB	celebrateBbucketBautobiographyBatticBatrocityBassetBadmirerBwatcherBwalmartB
unfinishedBthroneBtackleB	switchingBsubconsciousBsuaveBspectrumBsonnyB
somethingsBshamefulBshakesBserumBsaloonBrukhBrowlandsB	rosemarysBrisksBrewriteBrenoB	radiationBpreacherBpigsBpickfordB
photographBorphanBmuslimsBmidwayBmiddleclassBmiceBmanicB	magazinesBlistsBjuiceBjudithBit´sBinterspersedBinsanelyBimitateB	hungarianBhostsBhooperBhooksBhairyBgrinBgraffitiBflockBfayeBfalconBeyedB
encouragedB
elementaryBdisappearanceBdevelopmentsBdernBdarlingBdangersB
conflictedBcoincidencesBclosureBclinicBchipsBchipBcharltonBcecilBcampfireBbyeBbrickB
braveheartBblaxploitationBbimboBberkeleyBbcBbaldB	admirablyBabsorbedBwolvesB	unwillingBtorchBtillyBthunderbirdsB
temptationB
supportiveBstirringBspringsBsniperBskippingBsemiBrepresentativeBreneBrelatesBregisterBredgraveBquantumBpunsBperilBpennedBpairedB
marvellousB	lingeringBlesbiansBlengthsBkissesBkingsleyBkathleenBjokerBjanesB
identitiesBharrietBgreeneBglossyBforrestB
flamboyantBfannyB	fairbanksBestherBelliotBdriftBdishB	defendingBdeborahBdashBdamBcommendableB
chupacabraBchucklesBchopsB	chocolateBchewBceremonyBbombedB
beforehandBbackedB	animatorsBamidBaerialBwronglyBvirtueBveronicaBtrumanBtormentB
talentlessB	superstarBstraighttovideoBstapleBspinningBspawnedBsorrowBslipsBslaveryBslaughteredBshelvesBsharksBsearchedBscorpionB	rooseveltBritchieBreunitedBretroBrecipeBrecallsBrapesBpsychiatricB	predictedBposeyBpolarBpistolB	permanentBperkinsBpenB	pattersonB	passengerB	paragraphB	overnightBoutlawBorientedB	operatingBoccultBobservedBobsceneBnotingB	motivatedB
mercifullyB	macdonaldBlovinglyBlinearBlandmarkBkristinBkiteBkinkyBkindnessBkeatonsBkaufmanBkatieBjacobBinducingBimhoBhoustonBhanBgeniusesBgadgetsBfarewellBfamedBdrownedBdreamyB	distortedBdimBdeedsBdangerouslyBcrowdsB	conveyingB
continuousB	condemnedB	competingB
classifiedBchoreBchopperB	childlikeB	cancelledBbrynnerB	bodyguardBballoonBartisticallyBamberBadvisedBadventurousB	accompanyByearningB	violentlyBvinnieB
uncreditedBtruthsBtrustedBtonedBtitsBthurmanBtheodoreBtalkyBsunkB
structuredBsirkBshredBshoutBshoBseattleBseagalsBschemesB	sasquatchBruledBrkoBrepresentingB	rebellionBrandallBraidersBquartersBproneBpressedB	policemenBplaguedBparrotBoutrageouslyB	norwegianBnoelBmillandBmanagingBmaeBlightlyBlifelongB
investmentB	interplayBinstrumentsB
instructorB
imprisonedBhandyBgreaseBgovernmentsBgigBgeoffreyB	fishburneBfightersBexamineBduhBdrainBdirectorwriterB	dimwittedBdemeanorB	deceptionBdarcyBcubanB	creationsBcowboysBcountedBcoolestBcompassionateBcohesiveBclientBchoirBchemicalBchefBcarriageBcaligulaB	butterflyBburntBburkeB
boyfriendsBboardsBblessedBbitesBbikoBbastardBbanditBbachchanBawardedBavengeB	automaticBauraBattenboroughBannoyBanatomyBamateursBalotB
adequatelyByourBworryingB	whimsicalBwebbBwearyBvaryingBupstairsBtreatingBtouristsBtilBswansonBstanceBstakesBsoxBsoberBshearerBshadyBsealBsailorsBresurrectionB	reasoningBraunchyBramseyBradarBquintessentialBpranksBplacingBpianistBoutdoorBmonksBmoldBmaturityBlonBlogBloanBlizardB	lifeforceBlaserBkolchakBjaredBirresponsibleBincompetenceBhmmmBheroinesBgestureBgaspB
fulllengthBfiendBfantasticallyBeuropaB
enormouslyBduncanBdomainBdisappearingBdietrichBdestructiveBdependBdeannaBdarklyBdarioBcrookBcoworkerBcorrectnessBconnecticutBconcentrationB	communismBcodyBclientsBclaytonBcautionB
categoriesBcaptBbrandonBboyerB
borderlineBbeowulfBbashBbarberBarizonaBambianceB	agreementBvotingBvisitorBvincenzoB	unleashedBtrickedBtransparentB
thereafterBtaglineBswordsBswissBsupportsB
suggestiveBstoicBspontaneousBskateboardingBsheilaBschizophrenicBrevivalBretrieveBpuzzlingBpunksBpuertoB
protectingBpremisesBpatchBoperateB
occupationBnovelistBnigelBnailedBmoriartyBmishmashBluxuryBliamBlaunchedBjockBirwinB	inhabitedBinexperiencedBimpressionsBimplyBhilaryBharmonyBhairedB	functionsB
forebodingBfinishesBenduredB
encouragesBeleanorB
dreadfullyBdonBdogmaBdenverBdataBdafoeBcutterBcsiB
cronenbergBcheaperBcassieBbranchBboxesBboutBbleedingBbillieBbethBbergmansBbadnessBbackyardBathleticBarielBanguishBworshipBwhodB	weirdnessBwarfareBvulnerabilityBvillaB	videotapeBvertigoB	versatileBunravelB
tragicallyBthumbBtestedBtemperB	sylvesterBsophisticationBsmallestBslutBslumberBsitesB	showcasesBscoopBsaddestBroboticBroachB
rightfullyBreuniteBrathboneB	prevalentB
preferablyBpredictabilityBpixarBoweBoverwhelmedB	overactedBoctopusBnunsBnavalB
millenniumBmangaBmagicianBluridBlorenzoBlockeBlaurieBlauBkerrBkeeperBjoyceBjediBiconsBhuppertBhumBhugoBhiphopBgymBgreeceBgracesBgossipBgoshBgenieBfussBfreddysBfloydBfiascoBfashionsBextraordinarilyB	expressesBenthrallingBedithBearnsBdrowningB	documentsBdisconnectedBdexterBdelveBdefyBcrocBconstraintsB
consistingBcolemanBchewingBcheatsBcapsuleBcalB	butcheredBbicycleBbenjaminBbelieverBanxietyB	annoyanceBallstarBacclaimBwipeBvolumesBversaBvanillaBunremarkableB	unnervingB	uncertainBtripsB	torturingBtoolsBtendedBtempestBtediumB
sweetheartBstinkB
standpointBskippedBskatingBshovedBshepardBshamelesslyBsgB	sentencedBsantB	replacingBremorseBratsoBratioBrangingBrampantBramonBramBpsychologicallyBprayingBprankBpercentBpenisBoutputBnuanceBmockingB	misplacedBmerryBmeaningsBloonyBlombardBklausBjustificationBjokingBirresistibleBintelligentlyBintellectuallyB
illustrateBhunkBhersBherdBgoofsBgiggleBghostlyBfugitiveBfearedBfastforwardBfarscapeBexistentialB
everybodysB
entertainsB	emergencyBelusiveB
eliminatedBechoesBechoBeachBdruggedB	dismissedBdavidsBdahmerBcustomsBcurtainB
collectingBcoasterBcindyB
censorshipBcedricBcamillaBburstsBbrookeB	braindeadBbotchedB	bloodshedBbasketBbaffledBantiwarBantisemitismBacademicBwrathBwarpedBveronikaBvengefulBusageBuncoverBtruthfulBtriggerB	transformBthereofBtensionsB	stumblingB	strangestBsteamyBstarshipBsnapB	silvermanB	signatureBselleckBselfcenteredBsaintsB
sacrificedBrussBrobbingBriddenBrickmanBreviveBreeveBreeseBquaintB	promotionB	promotingBposseB	positionsBponderBpokeB
playwrightBpeaksB
oppositionBoddballB
neighboursBnapoleonBmondayBmockeryBmistyBmisterBminnelliBminersBmidgetBmentorBmcqueenBmazeBmauriceB
manipulateBlureBlonerBlimbsBlectureBjerkyBjerksBjanuaryBjaggerBironsBinterrogationB	intellectBinjectB	hypocrisyB	honorableBhensonBhealingBgoldblumBfreelyBflameBexorcismBestevezB	eliminateB
electronicB
earthquakeBdownbeatB
disciplineBdirkBdienBdesolateBdenyingBdemilleBdeepestB	crossfireBcoyoteBcorbettBconroyBconfinesBcommentatorsBcollaborationBcheungBchavezBcensorsBbulliesBbrunetteBbrentB
biologicalBbertB	bartenderBavenueB
atrocitiesBargentosBarchitectureBapproximatelyBanniversaryBamoralBallanBzealandByoursBwrestlemaniaB	wrenchingBwokeBwidowedBwarnersBvetBunluckyBunforgivableBthroatsBthelmaBtennisB
successionBstimulatingBstabbingBspearsBsolvingB
soderberghBsmackB
slowmotionB
shoestringBshaggyBsensibilitiesBscreenplaysBscheiderBscarletB	revoltingBrescuesB	reportersBrefugeBreformBreelsB
protectiveB	prolongedB	privilegeBprintsBpoitierBpenchantBparsonsBpaddedBmutedBmustacheB	moderndayB	miyazakisBminesB	mastersonBmarvinBledgerBjuddBitiBintruderB	interiorsBinsignificantBgratingBgrandeurB	geraldineBgentlyBfriendshipsBfieryBfascistB	fairytaleB
expressingB	exhaustedBepitomeBepisodicBearthsBdudesBdopeyBdivisionBdeneuveBdecemberBcryptBcrassBcosBcontinuouslyB	connivingBconcentratesB
companionsBcometB
classmatesBchloeBcannibalismBbrownsBbreakthroughBbrassBbondingB
associatesBarkinBappalledBairingBaccomplishmentBvisceralBunsuccessfulB	unnoticedB
undertakerBtraitB
tendernessBteamedBtackedBsyncBsuppliesBsupermarketBstaresBsqueezeBspacesBslugsB
skillfullyBsimplerBshoutsBseussBselectBsearchesBsackBrenownedBrelievedBraoulBrantingBpunishedBpricesB
prejudicesBpotentBpopeBplotlineBpattonBovertBoverseasBoutlookBoptionsB	opponentsBoperasBoneillB	offeringsB	occurringBmorseBmichelBmenuBmcdowellBmccarthyBmaintainingBlukasBkruegerBjollyBjewelryBhunkyBhowardsBhostelBholdenB	griffithsBgretaBgoslingBgiftsB	galacticaBfreddieBfleetingBfleetBfewerBfelliniB
expressiveBexpandedBeinsteinB	efficientBdiscountB	dependentB	democracyBdefiningB	consistedBconcreteBcomparesBcoenBclarenceBcircuitB
cheesinessBcarlyleBcagesBbuddingBblamesBbgradeBbehalfB	barbarianBbagsBateBastonishinglyBanitaBalarmBadulteryBadoredB	achievingByetiByarnBwitsBunratedBuninspiringBtrendyBtransvestiteBtimedBtepidBstripperBstirBsteppedBspectacularlyBsoylentBsnowyBsmarmyBshinyBschwarzeneggerBsampleBrumbleBruggedB	rodriguezBrichlyBrevolveBretainsBrabidBpulseBprospectB	priscillaBplugB
plantationBnotwithstandingBnoticingBnatashaBnaschyBmuscularB
mulhollandBmementoB
mastermindBmarvelouslyBmartianB
maintainedBmacBlevyBleapsBlawnBladBjarBintimacyBinterpretedBinnuendoB
incompleteBillustratesBhuttonBhustlerBhumiliationBhatingBgustoBgrannyBgaysBfundsBfundB	footstepsBfleeingB	firsttimeBfathomBexgirlfriendB	evocativeBellisBduchovnyBdrifterBdreamedBdopeBdominicBdolemiteBdistinguishB	disastersB
deservedlyBdemiBdeltaBdarthBcropB	corridorsBconvictsBconsistencyB	conductorBcompositionsB	complainsBclydeBcarlyBbumBbikersB
battlestarBbathtubBbadassB	attendantBashesBarmorBaptlyB	andersonsBallianceB	alexandreBzhangBwoefullyBwiresBwatchersBwaitsB
vaudevilleB
transplantBtossBthanBswiftB	surpassesBstitchesB
spielbergsBslowsB	shootoutsBsheridanBshahBscandalBroadsBrivalryBreliedBreiserBrecruitB
recreationBquigleyBprehistoricBprefersBplatformB	placementBpazBpausesBpassiveBolsenBoblivionBnovemberBnotionsB	nathanielB
monumentalB	monstrousBmaureenB	materialsBmagnificentlyBlyricalBlowbrowBloopBlayerBknackBjunkieBjamB
irrationalB	interpretB	inheritedB	indulgentBimplicationsB	imitatingBidiocyBhabitsBguidanceBgruffBgripsBgriffinBgrandparentsBgimmicksBgilmoreBfrenzyB	forgivingB
fictitiousBfeistyBemotionlessBdutiesBdrippingBdivingBdiscernibleB
devastatedBdenialBdasBdaftBcrispinB	costumingBconfederateBconductB
compromiseBcomBcolonyBcollarBclimateBcivilianBcheerleaderBcatwomanBcarrotBcaroleBbritneyB	botheringB	blackmailBbigfootBbeersBbaxterBbaddieB	apologizeBapocalypticB
animationsBaimsByouveBwendigoBwarningsBvisitorsBuncompromisingBtwodimensionalB	throwawayBthankfulBtestsBsydowBswitzerlandB	suspendedB	surpassedBsunriseBstereoBstabsBsparedB
sophomoricBsolomonBsocksBslobBslippedBslamBshrinkBshenanigansBshebaB	seductionBsatisfactoryBrollercoasterBripoffsBripleyB	rejectionBrapingB	punchlineBpukeBprofessionallyBpredicamentB
powerfullyB	positivesBpodBpepperBpatternsBpartialBovertlyBounceBoptimismB	northwestBmotifBmormonsBmolB	marriagesBmarlowB	mandatoryBmackBmabelBlinksBliliB	lettermanBlaraBkleinBjigsawBjessieBiranianB
invincibleBinvestB
instrumentB
infidelityBilBidaB	honeymoonBhailBhaggardBgrosslyBgrandpaBgoddessBgeneticBfilesB
fassbinderBentertainerBenforcementBdistastefulBdieselBdeviousBdevBdeniseBdeclaredBdeckB
cunninghamBcuesBcongoBcockneyB
clevernessBcarpetB	caribbeanB	cannibalsBbyrneBburyBbranaghsBblurredBblissBbillionBbehavesBbeauBbarnBatwillB
attributesBariseBanilBaltmansBalluringBallegoryBalainBadeleBworriesBwildlifeBwieldingBwhipBwheelsBwahlbergBvaultB
upbringingB	ultimatumB	tragediesBtinBtideBthreatsBthereinBtemporarilyBsweeneyB	struggledBstereotypingBstatureBstampB
stalingradBsopranoBslotBsiegeBshiftingBshakespeareanBshaftBseventhBsalvageBsacredBruntimeBrottingBrobberBrhymeBrestoreBrefusingBrefugeeB	recoveredB	ravishingBrackBquirksBprolificBpresumeBpresidentialB	premieredB	predatorsBphrasesBpeggBpatheticallyB
needlesslyBmythsBmutantsBmuscleBmossB
moderatelyBmendesBmeanspiritedBmathieuBmashBmartiansBmarisaB	marijuanaBloudlyBlocaleBliftsBlavaBknivesBkilljoyBitvBintendsB	indicatesB
hitchhikerBhiringBheartilyBgwynethBgroundedBgoersB	fulfilledBfreedBfreakedBformerlyBfabricBeyebrowsBexistentBexaminedB
enthralledBemploysBduringBdungeonBdrasticallyBdosesBdisdainB
diabolicalB	deliriousBdarkestBcrummyBcrackedBcockyBcherBcampersBcainBbynesBbunuelBbsgB
breathlessBblendsB	bickeringBbergenBbavaBbathingBamokBaimlessBactionpackedBzizekByulBwwiBwelcomedBvolcanoBvivaBturtleBtropicalBtinaBtheirBterminalB	temporaryBteaseBsymphonyB	surroundsB	supremelyB
supportersBstunkBstrawBstillsBstarvingBspelledBsomberBsnippetsBslangBshieldBshaunB	shatteredBscumBsammoBsabotageBrushesB	requisiteBrelaxingB	rehearsalB	recogniseBproudlyB
pronouncedBpreventsB	practicesBpovBpondBpoliteBpetersonB
perversionBpassionsBpalpableBopusBnuttyBmonstrosityBmonotoneBmisfitsBmidlerB	melbourneBlubitschBlocksBlexBladenBkomodoBkeiraB	jeffersonB
infectiousBimpersonationBhumiliatingBhomeworkBhicksBhgBheritageB	heartlessBhaventBhappierBhannibalBgarneredB
fulfillingBfamiliarityBexternalBexposingBenhancesBembarkBdwightBdiscussionsBdiegoBdebacleBdaleBcursingBcrowBcringingBcowroteBcountrysBconnieB	concludesBcompoundB	colourfulB	cleopatraB	civilizedBcharleyB	carnosaurBburdenBbumpBbritsBbridesBbafflingB
assortmentBassistBannoysBangieBanalyzeB
alienationBadministrationBzoomByoullB	willinglyBweavesBvirtuesBuniquelyB
unemployedBundevelopedBturgidBtraveledB
transcendsBtonysBtearingBtastyBtanksB	summarizeBstrayBstalksB
staggeringBstacyBsororityBsoreBsordidBslewBshirtsBseasBrootedBrangesBprueB
projectionB	projectedBpreparationB
postmodernBpostapocalypticBpicturesqueBpervertBpaulineBowlBoutsBneutralBmommaBmissilesB	miniatureBmeekBmanosBmanneredBlocalesBlineupBlatinoBjoanneB
jacquelineBilluminatedBhugBhispanicBhighlightedBhawaiiBhatchB	hardshipsBhaltBgrainB	goodnightBgaloreBfurB
formidableBforeheadBexpandBequalsBenvironmentalB	emptinessBeditsB
delusionalBdecayBcypherBcynthiaB
creepinessBconnorsBconnorBclawBcircumstanceBchillerBchibaBcenaBcarrollBbritBboatsBbarrierB	awkwardlyB
attributedBashBarnieBamiableBalliedB	alejandroBagathaB	addressesBactorsactressesBabbeyBwynorskiBwitlessBwillsBwidowerBvoyagerB
voiceoversB	virginityBventuraB	upsettingBuniversallyB
undeniableB	twentiethBtuckerB
travellingB
subtletiesB	stylisticBstubbornBsteppingB
speechlessBsmashingB
slowmovingBslashBshapesBsfxBsciencefictionBrockedBripperBringingBridersBrevolverBretainBregretsB
recoveringB	receptionBrailwayBraeB
questionedBpremB	pregnancyBprecededBphantasmBowningBoutsiderBoutbreakB
orchestralBoprahBninetiesB
newspapersBnarcissisticBmovieandBmoranBmontagesBmoleBmiaBmelvynBmeltBlutherBjoshuaBinsectsBindifferenceBideologyBhurtingBhonourBheydayBhallucinationsBgrammarBgoodnaturedBgodardBgershwinBgainsBfleesB
fitzgeraldBfanaticsBfallonBextremesBexcessesBexaggerationBevolveBemilioBeliBducklingBdublinBdrummerBdresslerBdrainedBdeformedBdakotaBconcordeBconcentratedBcolB	clockworkB
cigarettesBcargoBcapshawB
capitalismBbrushBbraveryB	bloodbathBblessingB
babysitterBatorB
articulateB
annoyinglyBanistonBalbumsBadaptingBacknowledgedBablyByokaiBwelchB
waterfrontBvenusBvalerieBuntilB
untalentedBunhingedBtroupeBtrickyBtomeiBtiringBtextureBsuzanneBstupidlyB
stepfatherB	staircaseBspittingBsparseBspadesBsnatchBslayerBsiblingBshrillB	setpiecesBsecludedB
seamlesslyBscoringBscamBrevisitBrestlessBregainB
reflectingBreeksBrecruitsBprintedBpokingBplodsBpicnicBpermanentlyBpeasantBpartyingBorganicBneglectBminnieBmathBmatchingBmaritalB
marginallyB
managementBmadlyBlogoBlockerBlistingBlearBlaputaBlamBkeitelBkazanBirvingBinheritanceBillustratedBhungerBholidaysBherringsBgungaBguidedB	glorifiedBgardnerBfutileBfinnishB
featuretteBfarcicalBexceedinglyBemployB	empathizeB
emmanuelleBeltonBelevateBdownwardB
dimensionsBdescendsBdelonBdamselBculminatingBcrewsB	cowrittenB	convertedB
connectingBcoincidentallyBchongBcherryBchansBchainedBcarefreeBcapoteBbrigitteBbloatedBblendingBbigotryBbarnesB
astonishedBashtonBarchiveB	arbitraryBannualBallisonB	adversityBadoptB
accustomedBzorroByupByaddaBwilsonsBwbBwaltzBvesselBvargasB
unresolvedBufoBtreyBtowBtoppedBtollBthreatenBthemedBtextbookB	tenaciousBtakashiB
suspicionsB	superiorsB	sprinkledBspellsB	smugglingBsmBslaterBsixteenB
separationBseducedBscreamedBscoutBschtickBrolandBrockneBrisky
??
Const_5Const*
_output_shapes	
:?N*
dtype0	*??
value??B??	?N"??                                                 	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?                                                              	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?       	      	      	      	      	      	      	      	      	      		      
	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	       	      !	      "	      #	      $	      %	      &	      '	      (	      )	      *	      +	      ,	      -	      .	      /	      0	      1	      2	      3	      4	      5	      6	      7	      8	      9	      :	      ;	      <	      =	      >	      ?	      @	      A	      B	      C	      D	      E	      F	      G	      H	      I	      J	      K	      L	      M	      N	      O	      P	      Q	      R	      S	      T	      U	      V	      W	      X	      Y	      Z	      [	      \	      ]	      ^	      _	      `	      a	      b	      c	      d	      e	      f	      g	      h	      i	      j	      k	      l	      m	      n	      o	      p	      q	      r	      s	      t	      u	      v	      w	      x	      y	      z	      {	      |	      }	      ~	      	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	       
      
      
      
      
      
      
      
      
      	
      

      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
       
      !
      "
      #
      $
      %
      &
      '
      (
      )
      *
      +
      ,
      -
      .
      /
      0
      1
      2
      3
      4
      5
      6
      7
      8
      9
      :
      ;
      <
      =
      >
      ?
      @
      A
      B
      C
      D
      E
      F
      G
      H
      I
      J
      K
      L
      M
      N
      O
      P
      Q
      R
      S
      T
      U
      V
      W
      X
      Y
      Z
      [
      \
      ]
      ^
      _
      `
      a
      b
      c
      d
      e
      f
      g
      h
      i
      j
      k
      l
      m
      n
      o
      p
      q
      r
      s
      t
      u
      v
      w
      x
      y
      z
      {
      |
      }
      ~
      
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                                      	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?        !      !      !      !      !      !      !      !      !      	!      
!      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !       !      !!      "!      #!      $!      %!      &!      '!      (!      )!      *!      +!      ,!      -!      .!      /!      0!      1!      2!      3!      4!      5!      6!      7!      8!      9!      :!      ;!      <!      =!      >!      ?!      @!      A!      B!      C!      D!      E!      F!      G!      H!      I!      J!      K!      L!      M!      N!      O!      P!      Q!      R!      S!      T!      U!      V!      W!      X!      Y!      Z!      [!      \!      ]!      ^!      _!      `!      a!      b!      c!      d!      e!      f!      g!      h!      i!      j!      k!      l!      m!      n!      o!      p!      q!      r!      s!      t!      u!      v!      w!      x!      y!      z!      {!      |!      }!      ~!      !      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!       "      "      "      "      "      "      "      "      "      	"      
"      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "       "      !"      ""      #"      $"      %"      &"      '"      ("      )"      *"      +"      ,"      -"      ."      /"      0"      1"      2"      3"      4"      5"      6"      7"      8"      9"      :"      ;"      <"      ="      >"      ?"      @"      A"      B"      C"      D"      E"      F"      G"      H"      I"      J"      K"      L"      M"      N"      O"      P"      Q"      R"      S"      T"      U"      V"      W"      X"      Y"      Z"      ["      \"      ]"      ^"      _"      `"      a"      b"      c"      d"      e"      f"      g"      h"      i"      j"      k"      l"      m"      n"      o"      p"      q"      r"      s"      t"      u"      v"      w"      x"      y"      z"      {"      |"      }"      ~"      "      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"       #      #      #      #      #      #      #      #      #      	#      
#      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #       #      !#      "#      ##      $#      %#      &#      '#      (#      )#      *#      +#      ,#      -#      .#      /#      0#      1#      2#      3#      4#      5#      6#      7#      8#      9#      :#      ;#      <#      =#      >#      ?#      @#      A#      B#      C#      D#      E#      F#      G#      H#      I#      J#      K#      L#      M#      N#      O#      P#      Q#      R#      S#      T#      U#      V#      W#      X#      Y#      Z#      [#      \#      ]#      ^#      _#      `#      a#      b#      c#      d#      e#      f#      g#      h#      i#      j#      k#      l#      m#      n#      o#      p#      q#      r#      s#      t#      u#      v#      w#      x#      y#      z#      {#      |#      }#      ~#      #      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#       $      $      $      $      $      $      $      $      $      	$      
$      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $       $      !$      "$      #$      $$      %$      &$      '$      ($      )$      *$      +$      ,$      -$      .$      /$      0$      1$      2$      3$      4$      5$      6$      7$      8$      9$      :$      ;$      <$      =$      >$      ?$      @$      A$      B$      C$      D$      E$      F$      G$      H$      I$      J$      K$      L$      M$      N$      O$      P$      Q$      R$      S$      T$      U$      V$      W$      X$      Y$      Z$      [$      \$      ]$      ^$      _$      `$      a$      b$      c$      d$      e$      f$      g$      h$      i$      j$      k$      l$      m$      n$      o$      p$      q$      r$      s$      t$      u$      v$      w$      x$      y$      z$      {$      |$      }$      ~$      $      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$       %      %      %      %      %      %      %      %      %      	%      
%      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %       %      !%      "%      #%      $%      %%      &%      '%      (%      )%      *%      +%      ,%      -%      .%      /%      0%      1%      2%      3%      4%      5%      6%      7%      8%      9%      :%      ;%      <%      =%      >%      ?%      @%      A%      B%      C%      D%      E%      F%      G%      H%      I%      J%      K%      L%      M%      N%      O%      P%      Q%      R%      S%      T%      U%      V%      W%      X%      Y%      Z%      [%      \%      ]%      ^%      _%      `%      a%      b%      c%      d%      e%      f%      g%      h%      i%      j%      k%      l%      m%      n%      o%      p%      q%      r%      s%      t%      u%      v%      w%      x%      y%      z%      {%      |%      }%      ~%      %      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%       &      &      &      &      &      &      &      &      &      	&      
&      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &       &      !&      "&      #&      $&      %&      &&      '&      (&      )&      *&      +&      ,&      -&      .&      /&      0&      1&      2&      3&      4&      5&      6&      7&      8&      9&      :&      ;&      <&      =&      >&      ?&      @&      A&      B&      C&      D&      E&      F&      G&      H&      I&      J&      K&      L&      M&      N&      O&      P&      Q&      R&      S&      T&      U&      V&      W&      X&      Y&      Z&      [&      \&      ]&      ^&      _&      `&      a&      b&      c&      d&      e&      f&      g&      h&      i&      j&      k&      l&      m&      n&      o&      p&      q&      r&      s&      t&      u&      v&      w&      x&      y&      z&      {&      |&      }&      ~&      &      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&       '      '      '      '      '      '      '      '      '      	'      
'      '      '      '      '      '      
?
StatefulPartitionedCallStatefulPartitionedCall
hash_tableConst_4Const_5*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *#
fR
__inference_<lambda>_79759
?
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *#
fR
__inference_<lambda>_79764
8
NoOpNoOp^PartitionedCall^StatefulPartitionedCall
?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2MutableHashTable*
Tkeys0*
Tvalues0	*#
_class
loc:@MutableHashTable*
_output_shapes

::
?8
Const_6Const"/device:CPU:0*
_output_shapes
: *
dtype0*?8
value?8B?8 B?8
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
;
	keras_api
_lookup_layer
_adapt_function*
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

embeddings*
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
?
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

&kernel
'bias*
?
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses
._random_generator* 
?
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses

5kernel
6bias*
'
1
&2
'3
54
65*
'
0
&1
'2
53
64*
* 
?
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
<trace_0
=trace_1
>trace_2
?trace_3* 
6
@trace_0
Atrace_1
Btrace_2
Ctrace_3* 
* 
?
Diter

Ebeta_1

Fbeta_2
	Gdecay
Hlearning_ratem?&m?'m?5m?6m?v?&v?'v?5v?6v?*

Iserving_default* 
* 
7
J	keras_api
Klookup_table
Ltoken_counts*

Mtrace_0* 

0*

0*
* 
?
Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Strace_0* 

Ttrace_0* 
hb
VARIABLE_VALUEembedding/embeddings:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
?
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

Ztrace_0* 

[trace_0* 

&0
'1*

&0
'1*
* 
?
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*

atrace_0* 

btrace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
?
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses* 

htrace_0
itrace_1* 

jtrace_0
ktrace_1* 
* 

50
61*

50
61*
* 
?
lnon_trainable_variables

mlayers
nmetrics
olayer_regularization_losses
player_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses*

qtrace_0* 

rtrace_0* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
.
0
1
2
3
4
5*

s0
t1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
R
u_initializer
v_create_resource
w_initialize
x_destroy_resource* 
?
y_create_resource
z_initialize
{_destroy_resource><layer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
|	variables
}	keras_api
	~total
	count*
M
?	variables
?	keras_api

?total

?count
?
_fn_kwargs*
* 

?trace_0* 

?trace_0* 

?trace_0* 

?trace_0* 

?trace_0* 

?trace_0* 

~0
1*

|	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
??
VARIABLE_VALUEAdam/embedding/embeddings/mVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/embedding/embeddings/vVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
(serving_default_text_vectorization_inputPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCall_1StatefulPartitionedCall(serving_default_text_vectorization_input
hash_tableConstConst_1Const_2embedding/embeddingsdense/kernel
dense/biasdense_1/kerneldense_1/bias*
Tin
2
		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
	*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_78808
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename(embedding/embeddings/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp?MutableHashTable_lookup_table_export_values/LookupTableExportV2AMutableHashTable_lookup_table_export_values/LookupTableExportV2:1total_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp/Adam/embedding/embeddings/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp/Adam/embedding/embeddings/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOpConst_6*'
Tin 
2		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_79873
?
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameembedding/embeddingsdense/kernel
dense/biasdense_1/kerneldense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateMutableHashTabletotal_1count_1totalcountAdam/embedding/embeddings/mAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/embedding/embeddings/vAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/v*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_79958??
?

?
*__inference_sequential_layer_call_fn_79083

inputs
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?N 
	unknown_4:  
	unknown_5: 
	unknown_6: 
	unknown_7:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
	*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_78237o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?
E__inference_sequential_layer_call_and_return_conditional_losses_78529
text_vectorization_inputO
Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	"
embedding_78513:	?N 
dense_78517:  
dense_78519: 
dense_1_78523: 
dense_1_78525:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?!embedding/StatefulPartitionedCall?>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2l
text_vectorization/StringLowerStringLowertext_vectorization_input*#
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*#
_output_shapes
:?????????*
pattern<br />*
rewrite ?
'text_vectorization/StaticRegexReplace_1StaticRegexReplace.text_vectorization/StaticRegexReplace:output:0*#
_output_shapes
:?????????*+
pattern \d+(?:\.\d*)?(?:[eE][+-]?\d+)?*
rewrite ?
'text_vectorization/StaticRegexReplace_2StaticRegexReplace0text_vectorization/StaticRegexReplace_1:output:0*#
_output_shapes
:?????????*
pattern@([A-Za-z0-9_]+)*
rewrite ?
'text_vectorization/StaticRegexReplace_3StaticRegexReplace0text_vectorization/StaticRegexReplace_2:output:0*#
_output_shapes
:?????????*
pattern	 which *
rewrite ?
'text_vectorization/StaticRegexReplace_4StaticRegexReplace0text_vectorization/StaticRegexReplace_3:output:0*#
_output_shapes
:?????????*
pattern
 couldn *
rewrite ?
'text_vectorization/StaticRegexReplace_5StaticRegexReplace0text_vectorization/StaticRegexReplace_4:output:0*#
_output_shapes
:?????????*
pattern once *
rewrite ?
'text_vectorization/StaticRegexReplace_6StaticRegexReplace0text_vectorization/StaticRegexReplace_5:output:0*#
_output_shapes
:?????????*
pattern is *
rewrite ?
'text_vectorization/StaticRegexReplace_7StaticRegexReplace0text_vectorization/StaticRegexReplace_6:output:0*#
_output_shapes
:?????????*
pattern on *
rewrite ?
'text_vectorization/StaticRegexReplace_8StaticRegexReplace0text_vectorization/StaticRegexReplace_7:output:0*#
_output_shapes
:?????????*
pattern some *
rewrite ?
'text_vectorization/StaticRegexReplace_9StaticRegexReplace0text_vectorization/StaticRegexReplace_8:output:0*#
_output_shapes
:?????????*
pattern not *
rewrite ?
(text_vectorization/StaticRegexReplace_10StaticRegexReplace0text_vectorization/StaticRegexReplace_9:output:0*#
_output_shapes
:?????????*
pattern won *
rewrite ?
(text_vectorization/StaticRegexReplace_11StaticRegexReplace1text_vectorization/StaticRegexReplace_10:output:0*#
_output_shapes
:?????????*
pattern	 while *
rewrite ?
(text_vectorization/StaticRegexReplace_12StaticRegexReplace1text_vectorization/StaticRegexReplace_11:output:0*#
_output_shapes
:?????????*
pattern them *
rewrite ?
(text_vectorization/StaticRegexReplace_13StaticRegexReplace1text_vectorization/StaticRegexReplace_12:output:0*#
_output_shapes
:?????????*
pattern am *
rewrite ?
(text_vectorization/StaticRegexReplace_14StaticRegexReplace1text_vectorization/StaticRegexReplace_13:output:0*#
_output_shapes
:?????????*
pattern	 where *
rewrite ?
(text_vectorization/StaticRegexReplace_15StaticRegexReplace1text_vectorization/StaticRegexReplace_14:output:0*#
_output_shapes
:?????????*
pattern my *
rewrite ?
(text_vectorization/StaticRegexReplace_16StaticRegexReplace1text_vectorization/StaticRegexReplace_15:output:0*#
_output_shapes
:?????????*
pattern me *
rewrite ?
(text_vectorization/StaticRegexReplace_17StaticRegexReplace1text_vectorization/StaticRegexReplace_16:output:0*#
_output_shapes
:?????????*
pattern
 couldn't *
rewrite ?
(text_vectorization/StaticRegexReplace_18StaticRegexReplace1text_vectorization/StaticRegexReplace_17:output:0*#
_output_shapes
:?????????*
pattern all *
rewrite ?
(text_vectorization/StaticRegexReplace_19StaticRegexReplace1text_vectorization/StaticRegexReplace_18:output:0*#
_output_shapes
:?????????*
pattern it's *
rewrite ?
(text_vectorization/StaticRegexReplace_20StaticRegexReplace1text_vectorization/StaticRegexReplace_19:output:0*#
_output_shapes
:?????????*
pattern off *
rewrite ?
(text_vectorization/StaticRegexReplace_21StaticRegexReplace1text_vectorization/StaticRegexReplace_20:output:0*#
_output_shapes
:?????????*
pattern so *
rewrite ?
(text_vectorization/StaticRegexReplace_22StaticRegexReplace1text_vectorization/StaticRegexReplace_21:output:0*#
_output_shapes
:?????????*
pattern
 mightn *
rewrite ?
(text_vectorization/StaticRegexReplace_23StaticRegexReplace1text_vectorization/StaticRegexReplace_22:output:0*#
_output_shapes
:?????????*
pattern our *
rewrite ?
(text_vectorization/StaticRegexReplace_24StaticRegexReplace1text_vectorization/StaticRegexReplace_23:output:0*#
_output_shapes
:?????????*
pattern aren *
rewrite ?
(text_vectorization/StaticRegexReplace_25StaticRegexReplace1text_vectorization/StaticRegexReplace_24:output:0*#
_output_shapes
:?????????*
pattern	 won't *
rewrite ?
(text_vectorization/StaticRegexReplace_26StaticRegexReplace1text_vectorization/StaticRegexReplace_25:output:0*#
_output_shapes
:?????????*
pattern the *
rewrite ?
(text_vectorization/StaticRegexReplace_27StaticRegexReplace1text_vectorization/StaticRegexReplace_26:output:0*#
_output_shapes
:?????????*
pattern
 wasn't *
rewrite ?
(text_vectorization/StaticRegexReplace_28StaticRegexReplace1text_vectorization/StaticRegexReplace_27:output:0*#
_output_shapes
:?????????*
pattern just *
rewrite ?
(text_vectorization/StaticRegexReplace_29StaticRegexReplace1text_vectorization/StaticRegexReplace_28:output:0*#
_output_shapes
:?????????*
pattern
 myself *
rewrite ?
(text_vectorization/StaticRegexReplace_30StaticRegexReplace1text_vectorization/StaticRegexReplace_29:output:0*#
_output_shapes
:?????????*
pattern	 after *
rewrite ?
(text_vectorization/StaticRegexReplace_31StaticRegexReplace1text_vectorization/StaticRegexReplace_30:output:0*#
_output_shapes
:?????????*
pattern from *
rewrite ?
(text_vectorization/StaticRegexReplace_32StaticRegexReplace1text_vectorization/StaticRegexReplace_31:output:0*#
_output_shapes
:?????????*
pattern d *
rewrite ?
(text_vectorization/StaticRegexReplace_33StaticRegexReplace1text_vectorization/StaticRegexReplace_32:output:0*#
_output_shapes
:?????????*
pattern	 mustn *
rewrite ?
(text_vectorization/StaticRegexReplace_34StaticRegexReplace1text_vectorization/StaticRegexReplace_33:output:0*#
_output_shapes
:?????????*
pattern	 doesn't *
rewrite ?
(text_vectorization/StaticRegexReplace_35StaticRegexReplace1text_vectorization/StaticRegexReplace_34:output:0*#
_output_shapes
:?????????*
pattern did *
rewrite ?
(text_vectorization/StaticRegexReplace_36StaticRegexReplace1text_vectorization/StaticRegexReplace_35:output:0*#
_output_shapes
:?????????*
pattern what *
rewrite ?
(text_vectorization/StaticRegexReplace_37StaticRegexReplace1text_vectorization/StaticRegexReplace_36:output:0*#
_output_shapes
:?????????*
pattern in *
rewrite ?
(text_vectorization/StaticRegexReplace_38StaticRegexReplace1text_vectorization/StaticRegexReplace_37:output:0*#
_output_shapes
:?????????*
pattern out *
rewrite ?
(text_vectorization/StaticRegexReplace_39StaticRegexReplace1text_vectorization/StaticRegexReplace_38:output:0*#
_output_shapes
:?????????*
pattern than *
rewrite ?
(text_vectorization/StaticRegexReplace_40StaticRegexReplace1text_vectorization/StaticRegexReplace_39:output:0*#
_output_shapes
:?????????*
pattern to *
rewrite ?
(text_vectorization/StaticRegexReplace_41StaticRegexReplace1text_vectorization/StaticRegexReplace_40:output:0*#
_output_shapes
:?????????*
pattern	 because *
rewrite ?
(text_vectorization/StaticRegexReplace_42StaticRegexReplace1text_vectorization/StaticRegexReplace_41:output:0*#
_output_shapes
:?????????*
pattern too *
rewrite ?
(text_vectorization/StaticRegexReplace_43StaticRegexReplace1text_vectorization/StaticRegexReplace_42:output:0*#
_output_shapes
:?????????*
pattern here *
rewrite ?
(text_vectorization/StaticRegexReplace_44StaticRegexReplace1text_vectorization/StaticRegexReplace_43:output:0*#
_output_shapes
:?????????*
pattern ma *
rewrite ?
(text_vectorization/StaticRegexReplace_45StaticRegexReplace1text_vectorization/StaticRegexReplace_44:output:0*#
_output_shapes
:?????????*
pattern but *
rewrite ?
(text_vectorization/StaticRegexReplace_46StaticRegexReplace1text_vectorization/StaticRegexReplace_45:output:0*#
_output_shapes
:?????????*
pattern
 before *
rewrite ?
(text_vectorization/StaticRegexReplace_47StaticRegexReplace1text_vectorization/StaticRegexReplace_46:output:0*#
_output_shapes
:?????????*
pattern then *
rewrite ?
(text_vectorization/StaticRegexReplace_48StaticRegexReplace1text_vectorization/StaticRegexReplace_47:output:0*#
_output_shapes
:?????????*
pattern
 should *
rewrite ?
(text_vectorization/StaticRegexReplace_49StaticRegexReplace1text_vectorization/StaticRegexReplace_48:output:0*#
_output_shapes
:?????????*
pattern are *
rewrite ?
(text_vectorization/StaticRegexReplace_50StaticRegexReplace1text_vectorization/StaticRegexReplace_49:output:0*#
_output_shapes
:?????????*
pattern had *
rewrite ?
(text_vectorization/StaticRegexReplace_51StaticRegexReplace1text_vectorization/StaticRegexReplace_50:output:0*#
_output_shapes
:?????????*
pattern	 himself *
rewrite ?
(text_vectorization/StaticRegexReplace_52StaticRegexReplace1text_vectorization/StaticRegexReplace_51:output:0*#
_output_shapes
:?????????*
pattern you *
rewrite ?
(text_vectorization/StaticRegexReplace_53StaticRegexReplace1text_vectorization/StaticRegexReplace_52:output:0*#
_output_shapes
:?????????*
pattern
 yourself *
rewrite ?
(text_vectorization/StaticRegexReplace_54StaticRegexReplace1text_vectorization/StaticRegexReplace_53:output:0*#
_output_shapes
:?????????*
pattern	 through *
rewrite ?
(text_vectorization/StaticRegexReplace_55StaticRegexReplace1text_vectorization/StaticRegexReplace_54:output:0*#
_output_shapes
:?????????*
pattern hadn *
rewrite ?
(text_vectorization/StaticRegexReplace_56StaticRegexReplace1text_vectorization/StaticRegexReplace_55:output:0*#
_output_shapes
:?????????*
pattern does *
rewrite ?
(text_vectorization/StaticRegexReplace_57StaticRegexReplace1text_vectorization/StaticRegexReplace_56:output:0*#
_output_shapes
:?????????*
pattern m *
rewrite ?
(text_vectorization/StaticRegexReplace_58StaticRegexReplace1text_vectorization/StaticRegexReplace_57:output:0*#
_output_shapes
:?????????*
pattern ain *
rewrite ?
(text_vectorization/StaticRegexReplace_59StaticRegexReplace1text_vectorization/StaticRegexReplace_58:output:0*#
_output_shapes
:?????????*
pattern very *
rewrite ?
(text_vectorization/StaticRegexReplace_60StaticRegexReplace1text_vectorization/StaticRegexReplace_59:output:0*#
_output_shapes
:?????????*
pattern	 weren't *
rewrite ?
(text_vectorization/StaticRegexReplace_61StaticRegexReplace1text_vectorization/StaticRegexReplace_60:output:0*#
_output_shapes
:?????????*
pattern been *
rewrite ?
(text_vectorization/StaticRegexReplace_62StaticRegexReplace1text_vectorization/StaticRegexReplace_61:output:0*#
_output_shapes
:?????????*
pattern will *
rewrite ?
(text_vectorization/StaticRegexReplace_63StaticRegexReplace1text_vectorization/StaticRegexReplace_62:output:0*#
_output_shapes
:?????????*
pattern now *
rewrite ?
(text_vectorization/StaticRegexReplace_64StaticRegexReplace1text_vectorization/StaticRegexReplace_63:output:0*#
_output_shapes
:?????????*
pattern they *
rewrite ?
(text_vectorization/StaticRegexReplace_65StaticRegexReplace1text_vectorization/StaticRegexReplace_64:output:0*#
_output_shapes
:?????????*
pattern when *
rewrite ?
(text_vectorization/StaticRegexReplace_66StaticRegexReplace1text_vectorization/StaticRegexReplace_65:output:0*#
_output_shapes
:?????????*
pattern was *
rewrite ?
(text_vectorization/StaticRegexReplace_67StaticRegexReplace1text_vectorization/StaticRegexReplace_66:output:0*#
_output_shapes
:?????????*
pattern shouldn't *
rewrite ?
(text_vectorization/StaticRegexReplace_68StaticRegexReplace1text_vectorization/StaticRegexReplace_67:output:0*#
_output_shapes
:?????????*
pattern	 herself *
rewrite ?
(text_vectorization/StaticRegexReplace_69StaticRegexReplace1text_vectorization/StaticRegexReplace_68:output:0*#
_output_shapes
:?????????*
pattern	 above *
rewrite ?
(text_vectorization/StaticRegexReplace_70StaticRegexReplace1text_vectorization/StaticRegexReplace_69:output:0*#
_output_shapes
:?????????*
pattern why *
rewrite ?
(text_vectorization/StaticRegexReplace_71StaticRegexReplace1text_vectorization/StaticRegexReplace_70:output:0*#
_output_shapes
:?????????*
pattern her *
rewrite ?
(text_vectorization/StaticRegexReplace_72StaticRegexReplace1text_vectorization/StaticRegexReplace_71:output:0*#
_output_shapes
:?????????*
pattern same *
rewrite ?
(text_vectorization/StaticRegexReplace_73StaticRegexReplace1text_vectorization/StaticRegexReplace_72:output:0*#
_output_shapes
:?????????*
pattern
 having *
rewrite ?
(text_vectorization/StaticRegexReplace_74StaticRegexReplace1text_vectorization/StaticRegexReplace_73:output:0*#
_output_shapes
:?????????*
pattern	 yours *
rewrite ?
(text_vectorization/StaticRegexReplace_75StaticRegexReplace1text_vectorization/StaticRegexReplace_74:output:0*#
_output_shapes
:?????????*
pattern can *
rewrite ?
(text_vectorization/StaticRegexReplace_76StaticRegexReplace1text_vectorization/StaticRegexReplace_75:output:0*#
_output_shapes
:?????????*
pattern
 wouldn't *
rewrite ?
(text_vectorization/StaticRegexReplace_77StaticRegexReplace1text_vectorization/StaticRegexReplace_76:output:0*#
_output_shapes
:?????????*
pattern	 again *
rewrite ?
(text_vectorization/StaticRegexReplace_78StaticRegexReplace1text_vectorization/StaticRegexReplace_77:output:0*#
_output_shapes
:?????????*
pattern do *
rewrite ?
(text_vectorization/StaticRegexReplace_79StaticRegexReplace1text_vectorization/StaticRegexReplace_78:output:0*#
_output_shapes
:?????????*
pattern shan *
rewrite ?
(text_vectorization/StaticRegexReplace_80StaticRegexReplace1text_vectorization/StaticRegexReplace_79:output:0*#
_output_shapes
:?????????*
pattern	 she's *
rewrite ?
(text_vectorization/StaticRegexReplace_81StaticRegexReplace1text_vectorization/StaticRegexReplace_80:output:0*#
_output_shapes
:?????????*
pattern of *
rewrite ?
(text_vectorization/StaticRegexReplace_82StaticRegexReplace1text_vectorization/StaticRegexReplace_81:output:0*#
_output_shapes
:?????????*
pattern	 against *
rewrite ?
(text_vectorization/StaticRegexReplace_83StaticRegexReplace1text_vectorization/StaticRegexReplace_82:output:0*#
_output_shapes
:?????????*
pattern most *
rewrite ?
(text_vectorization/StaticRegexReplace_84StaticRegexReplace1text_vectorization/StaticRegexReplace_83:output:0*#
_output_shapes
:?????????*
pattern	 isn't *
rewrite ?
(text_vectorization/StaticRegexReplace_85StaticRegexReplace1text_vectorization/StaticRegexReplace_84:output:0*#
_output_shapes
:?????????*
pattern	 until *
rewrite ?
(text_vectorization/StaticRegexReplace_86StaticRegexReplace1text_vectorization/StaticRegexReplace_85:output:0*#
_output_shapes
:?????????*
pattern it *
rewrite ?
(text_vectorization/StaticRegexReplace_87StaticRegexReplace1text_vectorization/StaticRegexReplace_86:output:0*#
_output_shapes
:?????????*
pattern	 below *
rewrite ?
(text_vectorization/StaticRegexReplace_88StaticRegexReplace1text_vectorization/StaticRegexReplace_87:output:0*#
_output_shapes
:?????????*
pattern	 mustn't *
rewrite ?
(text_vectorization/StaticRegexReplace_89StaticRegexReplace1text_vectorization/StaticRegexReplace_88:output:0*#
_output_shapes
:?????????*
pattern by *
rewrite ?
(text_vectorization/StaticRegexReplace_90StaticRegexReplace1text_vectorization/StaticRegexReplace_89:output:0*#
_output_shapes
:?????????*
pattern didn *
rewrite ?
(text_vectorization/StaticRegexReplace_91StaticRegexReplace1text_vectorization/StaticRegexReplace_90:output:0*#
_output_shapes
:?????????*
pattern
 shan't *
rewrite ?
(text_vectorization/StaticRegexReplace_92StaticRegexReplace1text_vectorization/StaticRegexReplace_91:output:0*#
_output_shapes
:?????????*
pattern who *
rewrite ?
(text_vectorization/StaticRegexReplace_93StaticRegexReplace1text_vectorization/StaticRegexReplace_92:output:0*#
_output_shapes
:?????????*
pattern both *
rewrite ?
(text_vectorization/StaticRegexReplace_94StaticRegexReplace1text_vectorization/StaticRegexReplace_93:output:0*#
_output_shapes
:?????????*
pattern re *
rewrite ?
(text_vectorization/StaticRegexReplace_95StaticRegexReplace1text_vectorization/StaticRegexReplace_94:output:0*#
_output_shapes
:?????????*
pattern
 wouldn *
rewrite ?
(text_vectorization/StaticRegexReplace_96StaticRegexReplace1text_vectorization/StaticRegexReplace_95:output:0*#
_output_shapes
:?????????*
pattern his *
rewrite ?
(text_vectorization/StaticRegexReplace_97StaticRegexReplace1text_vectorization/StaticRegexReplace_96:output:0*#
_output_shapes
:?????????*
pattern ours *
rewrite ?
(text_vectorization/StaticRegexReplace_98StaticRegexReplace1text_vectorization/StaticRegexReplace_97:output:0*#
_output_shapes
:?????????*
pattern
 itself *
rewrite ?
(text_vectorization/StaticRegexReplace_99StaticRegexReplace1text_vectorization/StaticRegexReplace_98:output:0*#
_output_shapes
:?????????*
pattern don *
rewrite ?
)text_vectorization/StaticRegexReplace_100StaticRegexReplace1text_vectorization/StaticRegexReplace_99:output:0*#
_output_shapes
:?????????*
pattern	 about *
rewrite ?
)text_vectorization/StaticRegexReplace_101StaticRegexReplace2text_vectorization/StaticRegexReplace_100:output:0*#
_output_shapes
:?????????*
pattern o *
rewrite ?
)text_vectorization/StaticRegexReplace_102StaticRegexReplace2text_vectorization/StaticRegexReplace_101:output:0*#
_output_shapes
:?????????*
pattern
 during *
rewrite ?
)text_vectorization/StaticRegexReplace_103StaticRegexReplace2text_vectorization/StaticRegexReplace_102:output:0*#
_output_shapes
:?????????*
pattern whom *
rewrite ?
)text_vectorization/StaticRegexReplace_104StaticRegexReplace2text_vectorization/StaticRegexReplace_103:output:0*#
_output_shapes
:?????????*
pattern
 mightn't *
rewrite ?
)text_vectorization/StaticRegexReplace_105StaticRegexReplace2text_vectorization/StaticRegexReplace_104:output:0*#
_output_shapes
:?????????*
pattern
 didn't *
rewrite ?
)text_vectorization/StaticRegexReplace_106StaticRegexReplace2text_vectorization/StaticRegexReplace_105:output:0*#
_output_shapes
:?????????*
pattern themselves *
rewrite ?
)text_vectorization/StaticRegexReplace_107StaticRegexReplace2text_vectorization/StaticRegexReplace_106:output:0*#
_output_shapes
:?????????*
pattern with *
rewrite ?
)text_vectorization/StaticRegexReplace_108StaticRegexReplace2text_vectorization/StaticRegexReplace_107:output:0*#
_output_shapes
:?????????*
pattern
 theirs *
rewrite ?
)text_vectorization/StaticRegexReplace_109StaticRegexReplace2text_vectorization/StaticRegexReplace_108:output:0*#
_output_shapes
:?????????*
pattern	 further *
rewrite ?
)text_vectorization/StaticRegexReplace_110StaticRegexReplace2text_vectorization/StaticRegexReplace_109:output:0*#
_output_shapes
:?????????*
pattern be *
rewrite ?
)text_vectorization/StaticRegexReplace_111StaticRegexReplace2text_vectorization/StaticRegexReplace_110:output:0*#
_output_shapes
:?????????*
pattern	 weren *
rewrite ?
)text_vectorization/StaticRegexReplace_112StaticRegexReplace2text_vectorization/StaticRegexReplace_111:output:0*#
_output_shapes
:?????????*
pattern own *
rewrite ?
)text_vectorization/StaticRegexReplace_113StaticRegexReplace2text_vectorization/StaticRegexReplace_112:output:0*#
_output_shapes
:?????????*
pattern into *
rewrite ?
)text_vectorization/StaticRegexReplace_114StaticRegexReplace2text_vectorization/StaticRegexReplace_113:output:0*#
_output_shapes
:?????????*
pattern t *
rewrite ?
)text_vectorization/StaticRegexReplace_115StaticRegexReplace2text_vectorization/StaticRegexReplace_114:output:0*#
_output_shapes
:?????????*
pattern	 haven *
rewrite ?
)text_vectorization/StaticRegexReplace_116StaticRegexReplace2text_vectorization/StaticRegexReplace_115:output:0*#
_output_shapes
:?????????*
pattern	 there *
rewrite ?
)text_vectorization/StaticRegexReplace_117StaticRegexReplace2text_vectorization/StaticRegexReplace_116:output:0*#
_output_shapes
:?????????*
pattern yourselves *
rewrite ?
)text_vectorization/StaticRegexReplace_118StaticRegexReplace2text_vectorization/StaticRegexReplace_117:output:0*#
_output_shapes
:?????????*
pattern
 aren't *
rewrite ?
)text_vectorization/StaticRegexReplace_119StaticRegexReplace2text_vectorization/StaticRegexReplace_118:output:0*#
_output_shapes
:?????????*
pattern
 you'll *
rewrite ?
)text_vectorization/StaticRegexReplace_120StaticRegexReplace2text_vectorization/StaticRegexReplace_119:output:0*#
_output_shapes
:?????????*
pattern how *
rewrite ?
)text_vectorization/StaticRegexReplace_121StaticRegexReplace2text_vectorization/StaticRegexReplace_120:output:0*#
_output_shapes
:?????????*
pattern ourselves *
rewrite ?
)text_vectorization/StaticRegexReplace_122StaticRegexReplace2text_vectorization/StaticRegexReplace_121:output:0*#
_output_shapes
:?????????*
pattern an *
rewrite ?
)text_vectorization/StaticRegexReplace_123StaticRegexReplace2text_vectorization/StaticRegexReplace_122:output:0*#
_output_shapes
:?????????*
pattern	 don't *
rewrite ?
)text_vectorization/StaticRegexReplace_124StaticRegexReplace2text_vectorization/StaticRegexReplace_123:output:0*#
_output_shapes
:?????????*
pattern	 doing *
rewrite ?
)text_vectorization/StaticRegexReplace_125StaticRegexReplace2text_vectorization/StaticRegexReplace_124:output:0*#
_output_shapes
:?????????*
pattern more *
rewrite ?
)text_vectorization/StaticRegexReplace_126StaticRegexReplace2text_vectorization/StaticRegexReplace_125:output:0*#
_output_shapes
:?????????*
pattern each *
rewrite ?
)text_vectorization/StaticRegexReplace_127StaticRegexReplace2text_vectorization/StaticRegexReplace_126:output:0*#
_output_shapes
:?????????*
pattern we *
rewrite ?
)text_vectorization/StaticRegexReplace_128StaticRegexReplace2text_vectorization/StaticRegexReplace_127:output:0*#
_output_shapes
:?????????*
pattern	 these *
rewrite ?
)text_vectorization/StaticRegexReplace_129StaticRegexReplace2text_vectorization/StaticRegexReplace_128:output:0*#
_output_shapes
:?????????*
pattern over *
rewrite ?
)text_vectorization/StaticRegexReplace_130StaticRegexReplace2text_vectorization/StaticRegexReplace_129:output:0*#
_output_shapes
:?????????*
pattern i *
rewrite ?
)text_vectorization/StaticRegexReplace_131StaticRegexReplace2text_vectorization/StaticRegexReplace_130:output:0*#
_output_shapes
:?????????*
pattern nor *
rewrite ?
)text_vectorization/StaticRegexReplace_132StaticRegexReplace2text_vectorization/StaticRegexReplace_131:output:0*#
_output_shapes
:?????????*
pattern	 needn't *
rewrite ?
)text_vectorization/StaticRegexReplace_133StaticRegexReplace2text_vectorization/StaticRegexReplace_132:output:0*#
_output_shapes
:?????????*
pattern ll *
rewrite ?
)text_vectorization/StaticRegexReplace_134StaticRegexReplace2text_vectorization/StaticRegexReplace_133:output:0*#
_output_shapes
:?????????*
pattern	 between *
rewrite ?
)text_vectorization/StaticRegexReplace_135StaticRegexReplace2text_vectorization/StaticRegexReplace_134:output:0*#
_output_shapes
:?????????*
pattern should've *
rewrite ?
)text_vectorization/StaticRegexReplace_136StaticRegexReplace2text_vectorization/StaticRegexReplace_135:output:0*#
_output_shapes
:?????????*
pattern
 hadn't *
rewrite ?
)text_vectorization/StaticRegexReplace_137StaticRegexReplace2text_vectorization/StaticRegexReplace_136:output:0*#
_output_shapes
:?????????*
pattern hasn *
rewrite ?
)text_vectorization/StaticRegexReplace_138StaticRegexReplace2text_vectorization/StaticRegexReplace_137:output:0*#
_output_shapes
:?????????*
pattern were *
rewrite ?
)text_vectorization/StaticRegexReplace_139StaticRegexReplace2text_vectorization/StaticRegexReplace_138:output:0*#
_output_shapes
:?????????*
pattern has *
rewrite ?
)text_vectorization/StaticRegexReplace_140StaticRegexReplace2text_vectorization/StaticRegexReplace_139:output:0*#
_output_shapes
:?????????*
pattern only *
rewrite ?
)text_vectorization/StaticRegexReplace_141StaticRegexReplace2text_vectorization/StaticRegexReplace_140:output:0*#
_output_shapes
:?????????*
pattern she *
rewrite ?
)text_vectorization/StaticRegexReplace_142StaticRegexReplace2text_vectorization/StaticRegexReplace_141:output:0*#
_output_shapes
:?????????*
pattern	 needn *
rewrite ?
)text_vectorization/StaticRegexReplace_143StaticRegexReplace2text_vectorization/StaticRegexReplace_142:output:0*#
_output_shapes
:?????????*
pattern	 other *
rewrite ?
)text_vectorization/StaticRegexReplace_144StaticRegexReplace2text_vectorization/StaticRegexReplace_143:output:0*#
_output_shapes
:?????????*
pattern
 hasn't *
rewrite ?
)text_vectorization/StaticRegexReplace_145StaticRegexReplace2text_vectorization/StaticRegexReplace_144:output:0*#
_output_shapes
:?????????*
pattern a *
rewrite ?
)text_vectorization/StaticRegexReplace_146StaticRegexReplace2text_vectorization/StaticRegexReplace_145:output:0*#
_output_shapes
:?????????*
pattern	 shouldn *
rewrite ?
)text_vectorization/StaticRegexReplace_147StaticRegexReplace2text_vectorization/StaticRegexReplace_146:output:0*#
_output_shapes
:?????????*
pattern and *
rewrite ?
)text_vectorization/StaticRegexReplace_148StaticRegexReplace2text_vectorization/StaticRegexReplace_147:output:0*#
_output_shapes
:?????????*
pattern	 those *
rewrite ?
)text_vectorization/StaticRegexReplace_149StaticRegexReplace2text_vectorization/StaticRegexReplace_148:output:0*#
_output_shapes
:?????????*
pattern	 being *
rewrite ?
)text_vectorization/StaticRegexReplace_150StaticRegexReplace2text_vectorization/StaticRegexReplace_149:output:0*#
_output_shapes
:?????????*
pattern such *
rewrite ?
)text_vectorization/StaticRegexReplace_151StaticRegexReplace2text_vectorization/StaticRegexReplace_150:output:0*#
_output_shapes
:?????????*
pattern as *
rewrite ?
)text_vectorization/StaticRegexReplace_152StaticRegexReplace2text_vectorization/StaticRegexReplace_151:output:0*#
_output_shapes
:?????????*
pattern ve *
rewrite ?
)text_vectorization/StaticRegexReplace_153StaticRegexReplace2text_vectorization/StaticRegexReplace_152:output:0*#
_output_shapes
:?????????*
pattern hers *
rewrite ?
)text_vectorization/StaticRegexReplace_154StaticRegexReplace2text_vectorization/StaticRegexReplace_153:output:0*#
_output_shapes
:?????????*
pattern s *
rewrite ?
)text_vectorization/StaticRegexReplace_155StaticRegexReplace2text_vectorization/StaticRegexReplace_154:output:0*#
_output_shapes
:?????????*
pattern	 their *
rewrite ?
)text_vectorization/StaticRegexReplace_156StaticRegexReplace2text_vectorization/StaticRegexReplace_155:output:0*#
_output_shapes
:?????????*
pattern	 haven't *
rewrite ?
)text_vectorization/StaticRegexReplace_157StaticRegexReplace2text_vectorization/StaticRegexReplace_156:output:0*#
_output_shapes
:?????????*
pattern for *
rewrite ?
)text_vectorization/StaticRegexReplace_158StaticRegexReplace2text_vectorization/StaticRegexReplace_157:output:0*#
_output_shapes
:?????????*
pattern if *
rewrite ?
)text_vectorization/StaticRegexReplace_159StaticRegexReplace2text_vectorization/StaticRegexReplace_158:output:0*#
_output_shapes
:?????????*
pattern that *
rewrite ?
)text_vectorization/StaticRegexReplace_160StaticRegexReplace2text_vectorization/StaticRegexReplace_159:output:0*#
_output_shapes
:?????????*
pattern isn *
rewrite ?
)text_vectorization/StaticRegexReplace_161StaticRegexReplace2text_vectorization/StaticRegexReplace_160:output:0*#
_output_shapes
:?????????*
pattern him *
rewrite ?
)text_vectorization/StaticRegexReplace_162StaticRegexReplace2text_vectorization/StaticRegexReplace_161:output:0*#
_output_shapes
:?????????*
pattern wasn *
rewrite ?
)text_vectorization/StaticRegexReplace_163StaticRegexReplace2text_vectorization/StaticRegexReplace_162:output:0*#
_output_shapes
:?????????*
pattern any *
rewrite ?
)text_vectorization/StaticRegexReplace_164StaticRegexReplace2text_vectorization/StaticRegexReplace_163:output:0*#
_output_shapes
:?????????*
pattern have *
rewrite ?
)text_vectorization/StaticRegexReplace_165StaticRegexReplace2text_vectorization/StaticRegexReplace_164:output:0*#
_output_shapes
:?????????*
pattern	 under *
rewrite ?
)text_vectorization/StaticRegexReplace_166StaticRegexReplace2text_vectorization/StaticRegexReplace_165:output:0*#
_output_shapes
:?????????*
pattern	 that'll *
rewrite ?
)text_vectorization/StaticRegexReplace_167StaticRegexReplace2text_vectorization/StaticRegexReplace_166:output:0*#
_output_shapes
:?????????*
pattern or *
rewrite ?
)text_vectorization/StaticRegexReplace_168StaticRegexReplace2text_vectorization/StaticRegexReplace_167:output:0*#
_output_shapes
:?????????*
pattern no *
rewrite ?
)text_vectorization/StaticRegexReplace_169StaticRegexReplace2text_vectorization/StaticRegexReplace_168:output:0*#
_output_shapes
:?????????*
pattern he *
rewrite ?
)text_vectorization/StaticRegexReplace_170StaticRegexReplace2text_vectorization/StaticRegexReplace_169:output:0*#
_output_shapes
:?????????*
pattern
 you're *
rewrite ?
)text_vectorization/StaticRegexReplace_171StaticRegexReplace2text_vectorization/StaticRegexReplace_170:output:0*#
_output_shapes
:?????????*
pattern this *
rewrite ?
)text_vectorization/StaticRegexReplace_172StaticRegexReplace2text_vectorization/StaticRegexReplace_171:output:0*#
_output_shapes
:?????????*
pattern	 doesn *
rewrite ?
)text_vectorization/StaticRegexReplace_173StaticRegexReplace2text_vectorization/StaticRegexReplace_172:output:0*#
_output_shapes
:?????????*
pattern	 you'd *
rewrite ?
)text_vectorization/StaticRegexReplace_174StaticRegexReplace2text_vectorization/StaticRegexReplace_173:output:0*#
_output_shapes
:?????????*
pattern up *
rewrite ?
)text_vectorization/StaticRegexReplace_175StaticRegexReplace2text_vectorization/StaticRegexReplace_174:output:0*#
_output_shapes
:?????????*
pattern
 you've *
rewrite ?
)text_vectorization/StaticRegexReplace_176StaticRegexReplace2text_vectorization/StaticRegexReplace_175:output:0*#
_output_shapes
:?????????*
pattern your *
rewrite ?
)text_vectorization/StaticRegexReplace_177StaticRegexReplace2text_vectorization/StaticRegexReplace_176:output:0*#
_output_shapes
:?????????*
pattern at *
rewrite ?
)text_vectorization/StaticRegexReplace_178StaticRegexReplace2text_vectorization/StaticRegexReplace_177:output:0*#
_output_shapes
:?????????*
pattern few *
rewrite ?
)text_vectorization/StaticRegexReplace_179StaticRegexReplace2text_vectorization/StaticRegexReplace_178:output:0*#
_output_shapes
:?????????*
pattern its *
rewrite ?
)text_vectorization/StaticRegexReplace_180StaticRegexReplace2text_vectorization/StaticRegexReplace_179:output:0*#
_output_shapes
:?????????*
pattern y *
rewrite ?
)text_vectorization/StaticRegexReplace_181StaticRegexReplace2text_vectorization/StaticRegexReplace_180:output:0*#
_output_shapes
:?????????*
pattern down *
rewrite ?
)text_vectorization/StaticRegexReplace_182StaticRegexReplace2text_vectorization/StaticRegexReplace_181:output:0*#
_output_shapes
:?????????*A
pattern64[!"\#\$%\&'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}\~]*
rewrite e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV22text_vectorization/StaticRegexReplace_182:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tGtext_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????d       ?
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????d*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
!embedding/StatefulPartitionedCallStatefulPartitionedCall?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0embedding_78513*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_77846?
(global_average_pooling1d/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_77598?
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_78517dense_78519*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_77862?
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_77873?
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_78523dense_1_78525*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_77885w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^embedding/StatefulPartitionedCall?^text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2?
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:] Y
#
_output_shapes
:?????????
2
_user_specified_nametext_vectorization_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
C
'__inference_dropout_layer_call_fn_79650

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_77873`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
??
?
E__inference_sequential_layer_call_and_return_conditional_losses_79337

inputsO
Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	3
 embedding_embedding_lookup_79315:	?N 6
$dense_matmul_readvariableop_resource:  3
%dense_biasadd_readvariableop_resource: 8
&dense_1_matmul_readvariableop_resource: 5
'dense_1_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?embedding/embedding_lookup?>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2Z
text_vectorization/StringLowerStringLowerinputs*#
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*#
_output_shapes
:?????????*
pattern<br />*
rewrite ?
'text_vectorization/StaticRegexReplace_1StaticRegexReplace.text_vectorization/StaticRegexReplace:output:0*#
_output_shapes
:?????????*+
pattern \d+(?:\.\d*)?(?:[eE][+-]?\d+)?*
rewrite ?
'text_vectorization/StaticRegexReplace_2StaticRegexReplace0text_vectorization/StaticRegexReplace_1:output:0*#
_output_shapes
:?????????*
pattern@([A-Za-z0-9_]+)*
rewrite ?
'text_vectorization/StaticRegexReplace_3StaticRegexReplace0text_vectorization/StaticRegexReplace_2:output:0*#
_output_shapes
:?????????*
pattern	 which *
rewrite ?
'text_vectorization/StaticRegexReplace_4StaticRegexReplace0text_vectorization/StaticRegexReplace_3:output:0*#
_output_shapes
:?????????*
pattern
 couldn *
rewrite ?
'text_vectorization/StaticRegexReplace_5StaticRegexReplace0text_vectorization/StaticRegexReplace_4:output:0*#
_output_shapes
:?????????*
pattern once *
rewrite ?
'text_vectorization/StaticRegexReplace_6StaticRegexReplace0text_vectorization/StaticRegexReplace_5:output:0*#
_output_shapes
:?????????*
pattern is *
rewrite ?
'text_vectorization/StaticRegexReplace_7StaticRegexReplace0text_vectorization/StaticRegexReplace_6:output:0*#
_output_shapes
:?????????*
pattern on *
rewrite ?
'text_vectorization/StaticRegexReplace_8StaticRegexReplace0text_vectorization/StaticRegexReplace_7:output:0*#
_output_shapes
:?????????*
pattern some *
rewrite ?
'text_vectorization/StaticRegexReplace_9StaticRegexReplace0text_vectorization/StaticRegexReplace_8:output:0*#
_output_shapes
:?????????*
pattern not *
rewrite ?
(text_vectorization/StaticRegexReplace_10StaticRegexReplace0text_vectorization/StaticRegexReplace_9:output:0*#
_output_shapes
:?????????*
pattern won *
rewrite ?
(text_vectorization/StaticRegexReplace_11StaticRegexReplace1text_vectorization/StaticRegexReplace_10:output:0*#
_output_shapes
:?????????*
pattern	 while *
rewrite ?
(text_vectorization/StaticRegexReplace_12StaticRegexReplace1text_vectorization/StaticRegexReplace_11:output:0*#
_output_shapes
:?????????*
pattern them *
rewrite ?
(text_vectorization/StaticRegexReplace_13StaticRegexReplace1text_vectorization/StaticRegexReplace_12:output:0*#
_output_shapes
:?????????*
pattern am *
rewrite ?
(text_vectorization/StaticRegexReplace_14StaticRegexReplace1text_vectorization/StaticRegexReplace_13:output:0*#
_output_shapes
:?????????*
pattern	 where *
rewrite ?
(text_vectorization/StaticRegexReplace_15StaticRegexReplace1text_vectorization/StaticRegexReplace_14:output:0*#
_output_shapes
:?????????*
pattern my *
rewrite ?
(text_vectorization/StaticRegexReplace_16StaticRegexReplace1text_vectorization/StaticRegexReplace_15:output:0*#
_output_shapes
:?????????*
pattern me *
rewrite ?
(text_vectorization/StaticRegexReplace_17StaticRegexReplace1text_vectorization/StaticRegexReplace_16:output:0*#
_output_shapes
:?????????*
pattern
 couldn't *
rewrite ?
(text_vectorization/StaticRegexReplace_18StaticRegexReplace1text_vectorization/StaticRegexReplace_17:output:0*#
_output_shapes
:?????????*
pattern all *
rewrite ?
(text_vectorization/StaticRegexReplace_19StaticRegexReplace1text_vectorization/StaticRegexReplace_18:output:0*#
_output_shapes
:?????????*
pattern it's *
rewrite ?
(text_vectorization/StaticRegexReplace_20StaticRegexReplace1text_vectorization/StaticRegexReplace_19:output:0*#
_output_shapes
:?????????*
pattern off *
rewrite ?
(text_vectorization/StaticRegexReplace_21StaticRegexReplace1text_vectorization/StaticRegexReplace_20:output:0*#
_output_shapes
:?????????*
pattern so *
rewrite ?
(text_vectorization/StaticRegexReplace_22StaticRegexReplace1text_vectorization/StaticRegexReplace_21:output:0*#
_output_shapes
:?????????*
pattern
 mightn *
rewrite ?
(text_vectorization/StaticRegexReplace_23StaticRegexReplace1text_vectorization/StaticRegexReplace_22:output:0*#
_output_shapes
:?????????*
pattern our *
rewrite ?
(text_vectorization/StaticRegexReplace_24StaticRegexReplace1text_vectorization/StaticRegexReplace_23:output:0*#
_output_shapes
:?????????*
pattern aren *
rewrite ?
(text_vectorization/StaticRegexReplace_25StaticRegexReplace1text_vectorization/StaticRegexReplace_24:output:0*#
_output_shapes
:?????????*
pattern	 won't *
rewrite ?
(text_vectorization/StaticRegexReplace_26StaticRegexReplace1text_vectorization/StaticRegexReplace_25:output:0*#
_output_shapes
:?????????*
pattern the *
rewrite ?
(text_vectorization/StaticRegexReplace_27StaticRegexReplace1text_vectorization/StaticRegexReplace_26:output:0*#
_output_shapes
:?????????*
pattern
 wasn't *
rewrite ?
(text_vectorization/StaticRegexReplace_28StaticRegexReplace1text_vectorization/StaticRegexReplace_27:output:0*#
_output_shapes
:?????????*
pattern just *
rewrite ?
(text_vectorization/StaticRegexReplace_29StaticRegexReplace1text_vectorization/StaticRegexReplace_28:output:0*#
_output_shapes
:?????????*
pattern
 myself *
rewrite ?
(text_vectorization/StaticRegexReplace_30StaticRegexReplace1text_vectorization/StaticRegexReplace_29:output:0*#
_output_shapes
:?????????*
pattern	 after *
rewrite ?
(text_vectorization/StaticRegexReplace_31StaticRegexReplace1text_vectorization/StaticRegexReplace_30:output:0*#
_output_shapes
:?????????*
pattern from *
rewrite ?
(text_vectorization/StaticRegexReplace_32StaticRegexReplace1text_vectorization/StaticRegexReplace_31:output:0*#
_output_shapes
:?????????*
pattern d *
rewrite ?
(text_vectorization/StaticRegexReplace_33StaticRegexReplace1text_vectorization/StaticRegexReplace_32:output:0*#
_output_shapes
:?????????*
pattern	 mustn *
rewrite ?
(text_vectorization/StaticRegexReplace_34StaticRegexReplace1text_vectorization/StaticRegexReplace_33:output:0*#
_output_shapes
:?????????*
pattern	 doesn't *
rewrite ?
(text_vectorization/StaticRegexReplace_35StaticRegexReplace1text_vectorization/StaticRegexReplace_34:output:0*#
_output_shapes
:?????????*
pattern did *
rewrite ?
(text_vectorization/StaticRegexReplace_36StaticRegexReplace1text_vectorization/StaticRegexReplace_35:output:0*#
_output_shapes
:?????????*
pattern what *
rewrite ?
(text_vectorization/StaticRegexReplace_37StaticRegexReplace1text_vectorization/StaticRegexReplace_36:output:0*#
_output_shapes
:?????????*
pattern in *
rewrite ?
(text_vectorization/StaticRegexReplace_38StaticRegexReplace1text_vectorization/StaticRegexReplace_37:output:0*#
_output_shapes
:?????????*
pattern out *
rewrite ?
(text_vectorization/StaticRegexReplace_39StaticRegexReplace1text_vectorization/StaticRegexReplace_38:output:0*#
_output_shapes
:?????????*
pattern than *
rewrite ?
(text_vectorization/StaticRegexReplace_40StaticRegexReplace1text_vectorization/StaticRegexReplace_39:output:0*#
_output_shapes
:?????????*
pattern to *
rewrite ?
(text_vectorization/StaticRegexReplace_41StaticRegexReplace1text_vectorization/StaticRegexReplace_40:output:0*#
_output_shapes
:?????????*
pattern	 because *
rewrite ?
(text_vectorization/StaticRegexReplace_42StaticRegexReplace1text_vectorization/StaticRegexReplace_41:output:0*#
_output_shapes
:?????????*
pattern too *
rewrite ?
(text_vectorization/StaticRegexReplace_43StaticRegexReplace1text_vectorization/StaticRegexReplace_42:output:0*#
_output_shapes
:?????????*
pattern here *
rewrite ?
(text_vectorization/StaticRegexReplace_44StaticRegexReplace1text_vectorization/StaticRegexReplace_43:output:0*#
_output_shapes
:?????????*
pattern ma *
rewrite ?
(text_vectorization/StaticRegexReplace_45StaticRegexReplace1text_vectorization/StaticRegexReplace_44:output:0*#
_output_shapes
:?????????*
pattern but *
rewrite ?
(text_vectorization/StaticRegexReplace_46StaticRegexReplace1text_vectorization/StaticRegexReplace_45:output:0*#
_output_shapes
:?????????*
pattern
 before *
rewrite ?
(text_vectorization/StaticRegexReplace_47StaticRegexReplace1text_vectorization/StaticRegexReplace_46:output:0*#
_output_shapes
:?????????*
pattern then *
rewrite ?
(text_vectorization/StaticRegexReplace_48StaticRegexReplace1text_vectorization/StaticRegexReplace_47:output:0*#
_output_shapes
:?????????*
pattern
 should *
rewrite ?
(text_vectorization/StaticRegexReplace_49StaticRegexReplace1text_vectorization/StaticRegexReplace_48:output:0*#
_output_shapes
:?????????*
pattern are *
rewrite ?
(text_vectorization/StaticRegexReplace_50StaticRegexReplace1text_vectorization/StaticRegexReplace_49:output:0*#
_output_shapes
:?????????*
pattern had *
rewrite ?
(text_vectorization/StaticRegexReplace_51StaticRegexReplace1text_vectorization/StaticRegexReplace_50:output:0*#
_output_shapes
:?????????*
pattern	 himself *
rewrite ?
(text_vectorization/StaticRegexReplace_52StaticRegexReplace1text_vectorization/StaticRegexReplace_51:output:0*#
_output_shapes
:?????????*
pattern you *
rewrite ?
(text_vectorization/StaticRegexReplace_53StaticRegexReplace1text_vectorization/StaticRegexReplace_52:output:0*#
_output_shapes
:?????????*
pattern
 yourself *
rewrite ?
(text_vectorization/StaticRegexReplace_54StaticRegexReplace1text_vectorization/StaticRegexReplace_53:output:0*#
_output_shapes
:?????????*
pattern	 through *
rewrite ?
(text_vectorization/StaticRegexReplace_55StaticRegexReplace1text_vectorization/StaticRegexReplace_54:output:0*#
_output_shapes
:?????????*
pattern hadn *
rewrite ?
(text_vectorization/StaticRegexReplace_56StaticRegexReplace1text_vectorization/StaticRegexReplace_55:output:0*#
_output_shapes
:?????????*
pattern does *
rewrite ?
(text_vectorization/StaticRegexReplace_57StaticRegexReplace1text_vectorization/StaticRegexReplace_56:output:0*#
_output_shapes
:?????????*
pattern m *
rewrite ?
(text_vectorization/StaticRegexReplace_58StaticRegexReplace1text_vectorization/StaticRegexReplace_57:output:0*#
_output_shapes
:?????????*
pattern ain *
rewrite ?
(text_vectorization/StaticRegexReplace_59StaticRegexReplace1text_vectorization/StaticRegexReplace_58:output:0*#
_output_shapes
:?????????*
pattern very *
rewrite ?
(text_vectorization/StaticRegexReplace_60StaticRegexReplace1text_vectorization/StaticRegexReplace_59:output:0*#
_output_shapes
:?????????*
pattern	 weren't *
rewrite ?
(text_vectorization/StaticRegexReplace_61StaticRegexReplace1text_vectorization/StaticRegexReplace_60:output:0*#
_output_shapes
:?????????*
pattern been *
rewrite ?
(text_vectorization/StaticRegexReplace_62StaticRegexReplace1text_vectorization/StaticRegexReplace_61:output:0*#
_output_shapes
:?????????*
pattern will *
rewrite ?
(text_vectorization/StaticRegexReplace_63StaticRegexReplace1text_vectorization/StaticRegexReplace_62:output:0*#
_output_shapes
:?????????*
pattern now *
rewrite ?
(text_vectorization/StaticRegexReplace_64StaticRegexReplace1text_vectorization/StaticRegexReplace_63:output:0*#
_output_shapes
:?????????*
pattern they *
rewrite ?
(text_vectorization/StaticRegexReplace_65StaticRegexReplace1text_vectorization/StaticRegexReplace_64:output:0*#
_output_shapes
:?????????*
pattern when *
rewrite ?
(text_vectorization/StaticRegexReplace_66StaticRegexReplace1text_vectorization/StaticRegexReplace_65:output:0*#
_output_shapes
:?????????*
pattern was *
rewrite ?
(text_vectorization/StaticRegexReplace_67StaticRegexReplace1text_vectorization/StaticRegexReplace_66:output:0*#
_output_shapes
:?????????*
pattern shouldn't *
rewrite ?
(text_vectorization/StaticRegexReplace_68StaticRegexReplace1text_vectorization/StaticRegexReplace_67:output:0*#
_output_shapes
:?????????*
pattern	 herself *
rewrite ?
(text_vectorization/StaticRegexReplace_69StaticRegexReplace1text_vectorization/StaticRegexReplace_68:output:0*#
_output_shapes
:?????????*
pattern	 above *
rewrite ?
(text_vectorization/StaticRegexReplace_70StaticRegexReplace1text_vectorization/StaticRegexReplace_69:output:0*#
_output_shapes
:?????????*
pattern why *
rewrite ?
(text_vectorization/StaticRegexReplace_71StaticRegexReplace1text_vectorization/StaticRegexReplace_70:output:0*#
_output_shapes
:?????????*
pattern her *
rewrite ?
(text_vectorization/StaticRegexReplace_72StaticRegexReplace1text_vectorization/StaticRegexReplace_71:output:0*#
_output_shapes
:?????????*
pattern same *
rewrite ?
(text_vectorization/StaticRegexReplace_73StaticRegexReplace1text_vectorization/StaticRegexReplace_72:output:0*#
_output_shapes
:?????????*
pattern
 having *
rewrite ?
(text_vectorization/StaticRegexReplace_74StaticRegexReplace1text_vectorization/StaticRegexReplace_73:output:0*#
_output_shapes
:?????????*
pattern	 yours *
rewrite ?
(text_vectorization/StaticRegexReplace_75StaticRegexReplace1text_vectorization/StaticRegexReplace_74:output:0*#
_output_shapes
:?????????*
pattern can *
rewrite ?
(text_vectorization/StaticRegexReplace_76StaticRegexReplace1text_vectorization/StaticRegexReplace_75:output:0*#
_output_shapes
:?????????*
pattern
 wouldn't *
rewrite ?
(text_vectorization/StaticRegexReplace_77StaticRegexReplace1text_vectorization/StaticRegexReplace_76:output:0*#
_output_shapes
:?????????*
pattern	 again *
rewrite ?
(text_vectorization/StaticRegexReplace_78StaticRegexReplace1text_vectorization/StaticRegexReplace_77:output:0*#
_output_shapes
:?????????*
pattern do *
rewrite ?
(text_vectorization/StaticRegexReplace_79StaticRegexReplace1text_vectorization/StaticRegexReplace_78:output:0*#
_output_shapes
:?????????*
pattern shan *
rewrite ?
(text_vectorization/StaticRegexReplace_80StaticRegexReplace1text_vectorization/StaticRegexReplace_79:output:0*#
_output_shapes
:?????????*
pattern	 she's *
rewrite ?
(text_vectorization/StaticRegexReplace_81StaticRegexReplace1text_vectorization/StaticRegexReplace_80:output:0*#
_output_shapes
:?????????*
pattern of *
rewrite ?
(text_vectorization/StaticRegexReplace_82StaticRegexReplace1text_vectorization/StaticRegexReplace_81:output:0*#
_output_shapes
:?????????*
pattern	 against *
rewrite ?
(text_vectorization/StaticRegexReplace_83StaticRegexReplace1text_vectorization/StaticRegexReplace_82:output:0*#
_output_shapes
:?????????*
pattern most *
rewrite ?
(text_vectorization/StaticRegexReplace_84StaticRegexReplace1text_vectorization/StaticRegexReplace_83:output:0*#
_output_shapes
:?????????*
pattern	 isn't *
rewrite ?
(text_vectorization/StaticRegexReplace_85StaticRegexReplace1text_vectorization/StaticRegexReplace_84:output:0*#
_output_shapes
:?????????*
pattern	 until *
rewrite ?
(text_vectorization/StaticRegexReplace_86StaticRegexReplace1text_vectorization/StaticRegexReplace_85:output:0*#
_output_shapes
:?????????*
pattern it *
rewrite ?
(text_vectorization/StaticRegexReplace_87StaticRegexReplace1text_vectorization/StaticRegexReplace_86:output:0*#
_output_shapes
:?????????*
pattern	 below *
rewrite ?
(text_vectorization/StaticRegexReplace_88StaticRegexReplace1text_vectorization/StaticRegexReplace_87:output:0*#
_output_shapes
:?????????*
pattern	 mustn't *
rewrite ?
(text_vectorization/StaticRegexReplace_89StaticRegexReplace1text_vectorization/StaticRegexReplace_88:output:0*#
_output_shapes
:?????????*
pattern by *
rewrite ?
(text_vectorization/StaticRegexReplace_90StaticRegexReplace1text_vectorization/StaticRegexReplace_89:output:0*#
_output_shapes
:?????????*
pattern didn *
rewrite ?
(text_vectorization/StaticRegexReplace_91StaticRegexReplace1text_vectorization/StaticRegexReplace_90:output:0*#
_output_shapes
:?????????*
pattern
 shan't *
rewrite ?
(text_vectorization/StaticRegexReplace_92StaticRegexReplace1text_vectorization/StaticRegexReplace_91:output:0*#
_output_shapes
:?????????*
pattern who *
rewrite ?
(text_vectorization/StaticRegexReplace_93StaticRegexReplace1text_vectorization/StaticRegexReplace_92:output:0*#
_output_shapes
:?????????*
pattern both *
rewrite ?
(text_vectorization/StaticRegexReplace_94StaticRegexReplace1text_vectorization/StaticRegexReplace_93:output:0*#
_output_shapes
:?????????*
pattern re *
rewrite ?
(text_vectorization/StaticRegexReplace_95StaticRegexReplace1text_vectorization/StaticRegexReplace_94:output:0*#
_output_shapes
:?????????*
pattern
 wouldn *
rewrite ?
(text_vectorization/StaticRegexReplace_96StaticRegexReplace1text_vectorization/StaticRegexReplace_95:output:0*#
_output_shapes
:?????????*
pattern his *
rewrite ?
(text_vectorization/StaticRegexReplace_97StaticRegexReplace1text_vectorization/StaticRegexReplace_96:output:0*#
_output_shapes
:?????????*
pattern ours *
rewrite ?
(text_vectorization/StaticRegexReplace_98StaticRegexReplace1text_vectorization/StaticRegexReplace_97:output:0*#
_output_shapes
:?????????*
pattern
 itself *
rewrite ?
(text_vectorization/StaticRegexReplace_99StaticRegexReplace1text_vectorization/StaticRegexReplace_98:output:0*#
_output_shapes
:?????????*
pattern don *
rewrite ?
)text_vectorization/StaticRegexReplace_100StaticRegexReplace1text_vectorization/StaticRegexReplace_99:output:0*#
_output_shapes
:?????????*
pattern	 about *
rewrite ?
)text_vectorization/StaticRegexReplace_101StaticRegexReplace2text_vectorization/StaticRegexReplace_100:output:0*#
_output_shapes
:?????????*
pattern o *
rewrite ?
)text_vectorization/StaticRegexReplace_102StaticRegexReplace2text_vectorization/StaticRegexReplace_101:output:0*#
_output_shapes
:?????????*
pattern
 during *
rewrite ?
)text_vectorization/StaticRegexReplace_103StaticRegexReplace2text_vectorization/StaticRegexReplace_102:output:0*#
_output_shapes
:?????????*
pattern whom *
rewrite ?
)text_vectorization/StaticRegexReplace_104StaticRegexReplace2text_vectorization/StaticRegexReplace_103:output:0*#
_output_shapes
:?????????*
pattern
 mightn't *
rewrite ?
)text_vectorization/StaticRegexReplace_105StaticRegexReplace2text_vectorization/StaticRegexReplace_104:output:0*#
_output_shapes
:?????????*
pattern
 didn't *
rewrite ?
)text_vectorization/StaticRegexReplace_106StaticRegexReplace2text_vectorization/StaticRegexReplace_105:output:0*#
_output_shapes
:?????????*
pattern themselves *
rewrite ?
)text_vectorization/StaticRegexReplace_107StaticRegexReplace2text_vectorization/StaticRegexReplace_106:output:0*#
_output_shapes
:?????????*
pattern with *
rewrite ?
)text_vectorization/StaticRegexReplace_108StaticRegexReplace2text_vectorization/StaticRegexReplace_107:output:0*#
_output_shapes
:?????????*
pattern
 theirs *
rewrite ?
)text_vectorization/StaticRegexReplace_109StaticRegexReplace2text_vectorization/StaticRegexReplace_108:output:0*#
_output_shapes
:?????????*
pattern	 further *
rewrite ?
)text_vectorization/StaticRegexReplace_110StaticRegexReplace2text_vectorization/StaticRegexReplace_109:output:0*#
_output_shapes
:?????????*
pattern be *
rewrite ?
)text_vectorization/StaticRegexReplace_111StaticRegexReplace2text_vectorization/StaticRegexReplace_110:output:0*#
_output_shapes
:?????????*
pattern	 weren *
rewrite ?
)text_vectorization/StaticRegexReplace_112StaticRegexReplace2text_vectorization/StaticRegexReplace_111:output:0*#
_output_shapes
:?????????*
pattern own *
rewrite ?
)text_vectorization/StaticRegexReplace_113StaticRegexReplace2text_vectorization/StaticRegexReplace_112:output:0*#
_output_shapes
:?????????*
pattern into *
rewrite ?
)text_vectorization/StaticRegexReplace_114StaticRegexReplace2text_vectorization/StaticRegexReplace_113:output:0*#
_output_shapes
:?????????*
pattern t *
rewrite ?
)text_vectorization/StaticRegexReplace_115StaticRegexReplace2text_vectorization/StaticRegexReplace_114:output:0*#
_output_shapes
:?????????*
pattern	 haven *
rewrite ?
)text_vectorization/StaticRegexReplace_116StaticRegexReplace2text_vectorization/StaticRegexReplace_115:output:0*#
_output_shapes
:?????????*
pattern	 there *
rewrite ?
)text_vectorization/StaticRegexReplace_117StaticRegexReplace2text_vectorization/StaticRegexReplace_116:output:0*#
_output_shapes
:?????????*
pattern yourselves *
rewrite ?
)text_vectorization/StaticRegexReplace_118StaticRegexReplace2text_vectorization/StaticRegexReplace_117:output:0*#
_output_shapes
:?????????*
pattern
 aren't *
rewrite ?
)text_vectorization/StaticRegexReplace_119StaticRegexReplace2text_vectorization/StaticRegexReplace_118:output:0*#
_output_shapes
:?????????*
pattern
 you'll *
rewrite ?
)text_vectorization/StaticRegexReplace_120StaticRegexReplace2text_vectorization/StaticRegexReplace_119:output:0*#
_output_shapes
:?????????*
pattern how *
rewrite ?
)text_vectorization/StaticRegexReplace_121StaticRegexReplace2text_vectorization/StaticRegexReplace_120:output:0*#
_output_shapes
:?????????*
pattern ourselves *
rewrite ?
)text_vectorization/StaticRegexReplace_122StaticRegexReplace2text_vectorization/StaticRegexReplace_121:output:0*#
_output_shapes
:?????????*
pattern an *
rewrite ?
)text_vectorization/StaticRegexReplace_123StaticRegexReplace2text_vectorization/StaticRegexReplace_122:output:0*#
_output_shapes
:?????????*
pattern	 don't *
rewrite ?
)text_vectorization/StaticRegexReplace_124StaticRegexReplace2text_vectorization/StaticRegexReplace_123:output:0*#
_output_shapes
:?????????*
pattern	 doing *
rewrite ?
)text_vectorization/StaticRegexReplace_125StaticRegexReplace2text_vectorization/StaticRegexReplace_124:output:0*#
_output_shapes
:?????????*
pattern more *
rewrite ?
)text_vectorization/StaticRegexReplace_126StaticRegexReplace2text_vectorization/StaticRegexReplace_125:output:0*#
_output_shapes
:?????????*
pattern each *
rewrite ?
)text_vectorization/StaticRegexReplace_127StaticRegexReplace2text_vectorization/StaticRegexReplace_126:output:0*#
_output_shapes
:?????????*
pattern we *
rewrite ?
)text_vectorization/StaticRegexReplace_128StaticRegexReplace2text_vectorization/StaticRegexReplace_127:output:0*#
_output_shapes
:?????????*
pattern	 these *
rewrite ?
)text_vectorization/StaticRegexReplace_129StaticRegexReplace2text_vectorization/StaticRegexReplace_128:output:0*#
_output_shapes
:?????????*
pattern over *
rewrite ?
)text_vectorization/StaticRegexReplace_130StaticRegexReplace2text_vectorization/StaticRegexReplace_129:output:0*#
_output_shapes
:?????????*
pattern i *
rewrite ?
)text_vectorization/StaticRegexReplace_131StaticRegexReplace2text_vectorization/StaticRegexReplace_130:output:0*#
_output_shapes
:?????????*
pattern nor *
rewrite ?
)text_vectorization/StaticRegexReplace_132StaticRegexReplace2text_vectorization/StaticRegexReplace_131:output:0*#
_output_shapes
:?????????*
pattern	 needn't *
rewrite ?
)text_vectorization/StaticRegexReplace_133StaticRegexReplace2text_vectorization/StaticRegexReplace_132:output:0*#
_output_shapes
:?????????*
pattern ll *
rewrite ?
)text_vectorization/StaticRegexReplace_134StaticRegexReplace2text_vectorization/StaticRegexReplace_133:output:0*#
_output_shapes
:?????????*
pattern	 between *
rewrite ?
)text_vectorization/StaticRegexReplace_135StaticRegexReplace2text_vectorization/StaticRegexReplace_134:output:0*#
_output_shapes
:?????????*
pattern should've *
rewrite ?
)text_vectorization/StaticRegexReplace_136StaticRegexReplace2text_vectorization/StaticRegexReplace_135:output:0*#
_output_shapes
:?????????*
pattern
 hadn't *
rewrite ?
)text_vectorization/StaticRegexReplace_137StaticRegexReplace2text_vectorization/StaticRegexReplace_136:output:0*#
_output_shapes
:?????????*
pattern hasn *
rewrite ?
)text_vectorization/StaticRegexReplace_138StaticRegexReplace2text_vectorization/StaticRegexReplace_137:output:0*#
_output_shapes
:?????????*
pattern were *
rewrite ?
)text_vectorization/StaticRegexReplace_139StaticRegexReplace2text_vectorization/StaticRegexReplace_138:output:0*#
_output_shapes
:?????????*
pattern has *
rewrite ?
)text_vectorization/StaticRegexReplace_140StaticRegexReplace2text_vectorization/StaticRegexReplace_139:output:0*#
_output_shapes
:?????????*
pattern only *
rewrite ?
)text_vectorization/StaticRegexReplace_141StaticRegexReplace2text_vectorization/StaticRegexReplace_140:output:0*#
_output_shapes
:?????????*
pattern she *
rewrite ?
)text_vectorization/StaticRegexReplace_142StaticRegexReplace2text_vectorization/StaticRegexReplace_141:output:0*#
_output_shapes
:?????????*
pattern	 needn *
rewrite ?
)text_vectorization/StaticRegexReplace_143StaticRegexReplace2text_vectorization/StaticRegexReplace_142:output:0*#
_output_shapes
:?????????*
pattern	 other *
rewrite ?
)text_vectorization/StaticRegexReplace_144StaticRegexReplace2text_vectorization/StaticRegexReplace_143:output:0*#
_output_shapes
:?????????*
pattern
 hasn't *
rewrite ?
)text_vectorization/StaticRegexReplace_145StaticRegexReplace2text_vectorization/StaticRegexReplace_144:output:0*#
_output_shapes
:?????????*
pattern a *
rewrite ?
)text_vectorization/StaticRegexReplace_146StaticRegexReplace2text_vectorization/StaticRegexReplace_145:output:0*#
_output_shapes
:?????????*
pattern	 shouldn *
rewrite ?
)text_vectorization/StaticRegexReplace_147StaticRegexReplace2text_vectorization/StaticRegexReplace_146:output:0*#
_output_shapes
:?????????*
pattern and *
rewrite ?
)text_vectorization/StaticRegexReplace_148StaticRegexReplace2text_vectorization/StaticRegexReplace_147:output:0*#
_output_shapes
:?????????*
pattern	 those *
rewrite ?
)text_vectorization/StaticRegexReplace_149StaticRegexReplace2text_vectorization/StaticRegexReplace_148:output:0*#
_output_shapes
:?????????*
pattern	 being *
rewrite ?
)text_vectorization/StaticRegexReplace_150StaticRegexReplace2text_vectorization/StaticRegexReplace_149:output:0*#
_output_shapes
:?????????*
pattern such *
rewrite ?
)text_vectorization/StaticRegexReplace_151StaticRegexReplace2text_vectorization/StaticRegexReplace_150:output:0*#
_output_shapes
:?????????*
pattern as *
rewrite ?
)text_vectorization/StaticRegexReplace_152StaticRegexReplace2text_vectorization/StaticRegexReplace_151:output:0*#
_output_shapes
:?????????*
pattern ve *
rewrite ?
)text_vectorization/StaticRegexReplace_153StaticRegexReplace2text_vectorization/StaticRegexReplace_152:output:0*#
_output_shapes
:?????????*
pattern hers *
rewrite ?
)text_vectorization/StaticRegexReplace_154StaticRegexReplace2text_vectorization/StaticRegexReplace_153:output:0*#
_output_shapes
:?????????*
pattern s *
rewrite ?
)text_vectorization/StaticRegexReplace_155StaticRegexReplace2text_vectorization/StaticRegexReplace_154:output:0*#
_output_shapes
:?????????*
pattern	 their *
rewrite ?
)text_vectorization/StaticRegexReplace_156StaticRegexReplace2text_vectorization/StaticRegexReplace_155:output:0*#
_output_shapes
:?????????*
pattern	 haven't *
rewrite ?
)text_vectorization/StaticRegexReplace_157StaticRegexReplace2text_vectorization/StaticRegexReplace_156:output:0*#
_output_shapes
:?????????*
pattern for *
rewrite ?
)text_vectorization/StaticRegexReplace_158StaticRegexReplace2text_vectorization/StaticRegexReplace_157:output:0*#
_output_shapes
:?????????*
pattern if *
rewrite ?
)text_vectorization/StaticRegexReplace_159StaticRegexReplace2text_vectorization/StaticRegexReplace_158:output:0*#
_output_shapes
:?????????*
pattern that *
rewrite ?
)text_vectorization/StaticRegexReplace_160StaticRegexReplace2text_vectorization/StaticRegexReplace_159:output:0*#
_output_shapes
:?????????*
pattern isn *
rewrite ?
)text_vectorization/StaticRegexReplace_161StaticRegexReplace2text_vectorization/StaticRegexReplace_160:output:0*#
_output_shapes
:?????????*
pattern him *
rewrite ?
)text_vectorization/StaticRegexReplace_162StaticRegexReplace2text_vectorization/StaticRegexReplace_161:output:0*#
_output_shapes
:?????????*
pattern wasn *
rewrite ?
)text_vectorization/StaticRegexReplace_163StaticRegexReplace2text_vectorization/StaticRegexReplace_162:output:0*#
_output_shapes
:?????????*
pattern any *
rewrite ?
)text_vectorization/StaticRegexReplace_164StaticRegexReplace2text_vectorization/StaticRegexReplace_163:output:0*#
_output_shapes
:?????????*
pattern have *
rewrite ?
)text_vectorization/StaticRegexReplace_165StaticRegexReplace2text_vectorization/StaticRegexReplace_164:output:0*#
_output_shapes
:?????????*
pattern	 under *
rewrite ?
)text_vectorization/StaticRegexReplace_166StaticRegexReplace2text_vectorization/StaticRegexReplace_165:output:0*#
_output_shapes
:?????????*
pattern	 that'll *
rewrite ?
)text_vectorization/StaticRegexReplace_167StaticRegexReplace2text_vectorization/StaticRegexReplace_166:output:0*#
_output_shapes
:?????????*
pattern or *
rewrite ?
)text_vectorization/StaticRegexReplace_168StaticRegexReplace2text_vectorization/StaticRegexReplace_167:output:0*#
_output_shapes
:?????????*
pattern no *
rewrite ?
)text_vectorization/StaticRegexReplace_169StaticRegexReplace2text_vectorization/StaticRegexReplace_168:output:0*#
_output_shapes
:?????????*
pattern he *
rewrite ?
)text_vectorization/StaticRegexReplace_170StaticRegexReplace2text_vectorization/StaticRegexReplace_169:output:0*#
_output_shapes
:?????????*
pattern
 you're *
rewrite ?
)text_vectorization/StaticRegexReplace_171StaticRegexReplace2text_vectorization/StaticRegexReplace_170:output:0*#
_output_shapes
:?????????*
pattern this *
rewrite ?
)text_vectorization/StaticRegexReplace_172StaticRegexReplace2text_vectorization/StaticRegexReplace_171:output:0*#
_output_shapes
:?????????*
pattern	 doesn *
rewrite ?
)text_vectorization/StaticRegexReplace_173StaticRegexReplace2text_vectorization/StaticRegexReplace_172:output:0*#
_output_shapes
:?????????*
pattern	 you'd *
rewrite ?
)text_vectorization/StaticRegexReplace_174StaticRegexReplace2text_vectorization/StaticRegexReplace_173:output:0*#
_output_shapes
:?????????*
pattern up *
rewrite ?
)text_vectorization/StaticRegexReplace_175StaticRegexReplace2text_vectorization/StaticRegexReplace_174:output:0*#
_output_shapes
:?????????*
pattern
 you've *
rewrite ?
)text_vectorization/StaticRegexReplace_176StaticRegexReplace2text_vectorization/StaticRegexReplace_175:output:0*#
_output_shapes
:?????????*
pattern your *
rewrite ?
)text_vectorization/StaticRegexReplace_177StaticRegexReplace2text_vectorization/StaticRegexReplace_176:output:0*#
_output_shapes
:?????????*
pattern at *
rewrite ?
)text_vectorization/StaticRegexReplace_178StaticRegexReplace2text_vectorization/StaticRegexReplace_177:output:0*#
_output_shapes
:?????????*
pattern few *
rewrite ?
)text_vectorization/StaticRegexReplace_179StaticRegexReplace2text_vectorization/StaticRegexReplace_178:output:0*#
_output_shapes
:?????????*
pattern its *
rewrite ?
)text_vectorization/StaticRegexReplace_180StaticRegexReplace2text_vectorization/StaticRegexReplace_179:output:0*#
_output_shapes
:?????????*
pattern y *
rewrite ?
)text_vectorization/StaticRegexReplace_181StaticRegexReplace2text_vectorization/StaticRegexReplace_180:output:0*#
_output_shapes
:?????????*
pattern down *
rewrite ?
)text_vectorization/StaticRegexReplace_182StaticRegexReplace2text_vectorization/StaticRegexReplace_181:output:0*#
_output_shapes
:?????????*A
pattern64[!"\#\$%\&'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}\~]*
rewrite e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV22text_vectorization/StaticRegexReplace_182:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tGtext_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????d       ?
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????d*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
embedding/embedding_lookupResourceGather embedding_embedding_lookup_79315?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*3
_class)
'%loc:@embedding/embedding_lookup/79315*+
_output_shapes
:?????????d *
dtype0?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*3
_class)
'%loc:@embedding/embedding_lookup/79315*+
_output_shapes
:?????????d ?
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????d q
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
global_average_pooling1d/MeanMean.embedding/embedding_lookup/Identity_1:output:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:????????? ?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0?
dense/MatMulMatMul&global_average_pooling1d/Mean:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? \

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:????????? h
dropout/IdentityIdentitydense/Relu:activations:0*
T0*'
_output_shapes
:????????? ?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
dense_1/MatMulMatMuldropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^embedding/embedding_lookup?^text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup2?
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
a
B__inference_dropout_layer_call_and_return_conditional_losses_79672

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:????????? C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:????????? Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
B__inference_dense_1_layer_call_and_return_conditional_losses_79691

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
:
__inference__creator_79696
identity??
hash_tablel

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name3447*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
?
'__inference_dense_1_layer_call_fn_79681

inputs
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_77885o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
??
?
E__inference_sequential_layer_call_and_return_conditional_losses_78777
text_vectorization_inputO
Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	"
embedding_78761:	?N 
dense_78765:  
dense_78767: 
dense_1_78771: 
dense_1_78773:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!embedding/StatefulPartitionedCall?>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2l
text_vectorization/StringLowerStringLowertext_vectorization_input*#
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*#
_output_shapes
:?????????*
pattern<br />*
rewrite ?
'text_vectorization/StaticRegexReplace_1StaticRegexReplace.text_vectorization/StaticRegexReplace:output:0*#
_output_shapes
:?????????*+
pattern \d+(?:\.\d*)?(?:[eE][+-]?\d+)?*
rewrite ?
'text_vectorization/StaticRegexReplace_2StaticRegexReplace0text_vectorization/StaticRegexReplace_1:output:0*#
_output_shapes
:?????????*
pattern@([A-Za-z0-9_]+)*
rewrite ?
'text_vectorization/StaticRegexReplace_3StaticRegexReplace0text_vectorization/StaticRegexReplace_2:output:0*#
_output_shapes
:?????????*
pattern	 which *
rewrite ?
'text_vectorization/StaticRegexReplace_4StaticRegexReplace0text_vectorization/StaticRegexReplace_3:output:0*#
_output_shapes
:?????????*
pattern
 couldn *
rewrite ?
'text_vectorization/StaticRegexReplace_5StaticRegexReplace0text_vectorization/StaticRegexReplace_4:output:0*#
_output_shapes
:?????????*
pattern once *
rewrite ?
'text_vectorization/StaticRegexReplace_6StaticRegexReplace0text_vectorization/StaticRegexReplace_5:output:0*#
_output_shapes
:?????????*
pattern is *
rewrite ?
'text_vectorization/StaticRegexReplace_7StaticRegexReplace0text_vectorization/StaticRegexReplace_6:output:0*#
_output_shapes
:?????????*
pattern on *
rewrite ?
'text_vectorization/StaticRegexReplace_8StaticRegexReplace0text_vectorization/StaticRegexReplace_7:output:0*#
_output_shapes
:?????????*
pattern some *
rewrite ?
'text_vectorization/StaticRegexReplace_9StaticRegexReplace0text_vectorization/StaticRegexReplace_8:output:0*#
_output_shapes
:?????????*
pattern not *
rewrite ?
(text_vectorization/StaticRegexReplace_10StaticRegexReplace0text_vectorization/StaticRegexReplace_9:output:0*#
_output_shapes
:?????????*
pattern won *
rewrite ?
(text_vectorization/StaticRegexReplace_11StaticRegexReplace1text_vectorization/StaticRegexReplace_10:output:0*#
_output_shapes
:?????????*
pattern	 while *
rewrite ?
(text_vectorization/StaticRegexReplace_12StaticRegexReplace1text_vectorization/StaticRegexReplace_11:output:0*#
_output_shapes
:?????????*
pattern them *
rewrite ?
(text_vectorization/StaticRegexReplace_13StaticRegexReplace1text_vectorization/StaticRegexReplace_12:output:0*#
_output_shapes
:?????????*
pattern am *
rewrite ?
(text_vectorization/StaticRegexReplace_14StaticRegexReplace1text_vectorization/StaticRegexReplace_13:output:0*#
_output_shapes
:?????????*
pattern	 where *
rewrite ?
(text_vectorization/StaticRegexReplace_15StaticRegexReplace1text_vectorization/StaticRegexReplace_14:output:0*#
_output_shapes
:?????????*
pattern my *
rewrite ?
(text_vectorization/StaticRegexReplace_16StaticRegexReplace1text_vectorization/StaticRegexReplace_15:output:0*#
_output_shapes
:?????????*
pattern me *
rewrite ?
(text_vectorization/StaticRegexReplace_17StaticRegexReplace1text_vectorization/StaticRegexReplace_16:output:0*#
_output_shapes
:?????????*
pattern
 couldn't *
rewrite ?
(text_vectorization/StaticRegexReplace_18StaticRegexReplace1text_vectorization/StaticRegexReplace_17:output:0*#
_output_shapes
:?????????*
pattern all *
rewrite ?
(text_vectorization/StaticRegexReplace_19StaticRegexReplace1text_vectorization/StaticRegexReplace_18:output:0*#
_output_shapes
:?????????*
pattern it's *
rewrite ?
(text_vectorization/StaticRegexReplace_20StaticRegexReplace1text_vectorization/StaticRegexReplace_19:output:0*#
_output_shapes
:?????????*
pattern off *
rewrite ?
(text_vectorization/StaticRegexReplace_21StaticRegexReplace1text_vectorization/StaticRegexReplace_20:output:0*#
_output_shapes
:?????????*
pattern so *
rewrite ?
(text_vectorization/StaticRegexReplace_22StaticRegexReplace1text_vectorization/StaticRegexReplace_21:output:0*#
_output_shapes
:?????????*
pattern
 mightn *
rewrite ?
(text_vectorization/StaticRegexReplace_23StaticRegexReplace1text_vectorization/StaticRegexReplace_22:output:0*#
_output_shapes
:?????????*
pattern our *
rewrite ?
(text_vectorization/StaticRegexReplace_24StaticRegexReplace1text_vectorization/StaticRegexReplace_23:output:0*#
_output_shapes
:?????????*
pattern aren *
rewrite ?
(text_vectorization/StaticRegexReplace_25StaticRegexReplace1text_vectorization/StaticRegexReplace_24:output:0*#
_output_shapes
:?????????*
pattern	 won't *
rewrite ?
(text_vectorization/StaticRegexReplace_26StaticRegexReplace1text_vectorization/StaticRegexReplace_25:output:0*#
_output_shapes
:?????????*
pattern the *
rewrite ?
(text_vectorization/StaticRegexReplace_27StaticRegexReplace1text_vectorization/StaticRegexReplace_26:output:0*#
_output_shapes
:?????????*
pattern
 wasn't *
rewrite ?
(text_vectorization/StaticRegexReplace_28StaticRegexReplace1text_vectorization/StaticRegexReplace_27:output:0*#
_output_shapes
:?????????*
pattern just *
rewrite ?
(text_vectorization/StaticRegexReplace_29StaticRegexReplace1text_vectorization/StaticRegexReplace_28:output:0*#
_output_shapes
:?????????*
pattern
 myself *
rewrite ?
(text_vectorization/StaticRegexReplace_30StaticRegexReplace1text_vectorization/StaticRegexReplace_29:output:0*#
_output_shapes
:?????????*
pattern	 after *
rewrite ?
(text_vectorization/StaticRegexReplace_31StaticRegexReplace1text_vectorization/StaticRegexReplace_30:output:0*#
_output_shapes
:?????????*
pattern from *
rewrite ?
(text_vectorization/StaticRegexReplace_32StaticRegexReplace1text_vectorization/StaticRegexReplace_31:output:0*#
_output_shapes
:?????????*
pattern d *
rewrite ?
(text_vectorization/StaticRegexReplace_33StaticRegexReplace1text_vectorization/StaticRegexReplace_32:output:0*#
_output_shapes
:?????????*
pattern	 mustn *
rewrite ?
(text_vectorization/StaticRegexReplace_34StaticRegexReplace1text_vectorization/StaticRegexReplace_33:output:0*#
_output_shapes
:?????????*
pattern	 doesn't *
rewrite ?
(text_vectorization/StaticRegexReplace_35StaticRegexReplace1text_vectorization/StaticRegexReplace_34:output:0*#
_output_shapes
:?????????*
pattern did *
rewrite ?
(text_vectorization/StaticRegexReplace_36StaticRegexReplace1text_vectorization/StaticRegexReplace_35:output:0*#
_output_shapes
:?????????*
pattern what *
rewrite ?
(text_vectorization/StaticRegexReplace_37StaticRegexReplace1text_vectorization/StaticRegexReplace_36:output:0*#
_output_shapes
:?????????*
pattern in *
rewrite ?
(text_vectorization/StaticRegexReplace_38StaticRegexReplace1text_vectorization/StaticRegexReplace_37:output:0*#
_output_shapes
:?????????*
pattern out *
rewrite ?
(text_vectorization/StaticRegexReplace_39StaticRegexReplace1text_vectorization/StaticRegexReplace_38:output:0*#
_output_shapes
:?????????*
pattern than *
rewrite ?
(text_vectorization/StaticRegexReplace_40StaticRegexReplace1text_vectorization/StaticRegexReplace_39:output:0*#
_output_shapes
:?????????*
pattern to *
rewrite ?
(text_vectorization/StaticRegexReplace_41StaticRegexReplace1text_vectorization/StaticRegexReplace_40:output:0*#
_output_shapes
:?????????*
pattern	 because *
rewrite ?
(text_vectorization/StaticRegexReplace_42StaticRegexReplace1text_vectorization/StaticRegexReplace_41:output:0*#
_output_shapes
:?????????*
pattern too *
rewrite ?
(text_vectorization/StaticRegexReplace_43StaticRegexReplace1text_vectorization/StaticRegexReplace_42:output:0*#
_output_shapes
:?????????*
pattern here *
rewrite ?
(text_vectorization/StaticRegexReplace_44StaticRegexReplace1text_vectorization/StaticRegexReplace_43:output:0*#
_output_shapes
:?????????*
pattern ma *
rewrite ?
(text_vectorization/StaticRegexReplace_45StaticRegexReplace1text_vectorization/StaticRegexReplace_44:output:0*#
_output_shapes
:?????????*
pattern but *
rewrite ?
(text_vectorization/StaticRegexReplace_46StaticRegexReplace1text_vectorization/StaticRegexReplace_45:output:0*#
_output_shapes
:?????????*
pattern
 before *
rewrite ?
(text_vectorization/StaticRegexReplace_47StaticRegexReplace1text_vectorization/StaticRegexReplace_46:output:0*#
_output_shapes
:?????????*
pattern then *
rewrite ?
(text_vectorization/StaticRegexReplace_48StaticRegexReplace1text_vectorization/StaticRegexReplace_47:output:0*#
_output_shapes
:?????????*
pattern
 should *
rewrite ?
(text_vectorization/StaticRegexReplace_49StaticRegexReplace1text_vectorization/StaticRegexReplace_48:output:0*#
_output_shapes
:?????????*
pattern are *
rewrite ?
(text_vectorization/StaticRegexReplace_50StaticRegexReplace1text_vectorization/StaticRegexReplace_49:output:0*#
_output_shapes
:?????????*
pattern had *
rewrite ?
(text_vectorization/StaticRegexReplace_51StaticRegexReplace1text_vectorization/StaticRegexReplace_50:output:0*#
_output_shapes
:?????????*
pattern	 himself *
rewrite ?
(text_vectorization/StaticRegexReplace_52StaticRegexReplace1text_vectorization/StaticRegexReplace_51:output:0*#
_output_shapes
:?????????*
pattern you *
rewrite ?
(text_vectorization/StaticRegexReplace_53StaticRegexReplace1text_vectorization/StaticRegexReplace_52:output:0*#
_output_shapes
:?????????*
pattern
 yourself *
rewrite ?
(text_vectorization/StaticRegexReplace_54StaticRegexReplace1text_vectorization/StaticRegexReplace_53:output:0*#
_output_shapes
:?????????*
pattern	 through *
rewrite ?
(text_vectorization/StaticRegexReplace_55StaticRegexReplace1text_vectorization/StaticRegexReplace_54:output:0*#
_output_shapes
:?????????*
pattern hadn *
rewrite ?
(text_vectorization/StaticRegexReplace_56StaticRegexReplace1text_vectorization/StaticRegexReplace_55:output:0*#
_output_shapes
:?????????*
pattern does *
rewrite ?
(text_vectorization/StaticRegexReplace_57StaticRegexReplace1text_vectorization/StaticRegexReplace_56:output:0*#
_output_shapes
:?????????*
pattern m *
rewrite ?
(text_vectorization/StaticRegexReplace_58StaticRegexReplace1text_vectorization/StaticRegexReplace_57:output:0*#
_output_shapes
:?????????*
pattern ain *
rewrite ?
(text_vectorization/StaticRegexReplace_59StaticRegexReplace1text_vectorization/StaticRegexReplace_58:output:0*#
_output_shapes
:?????????*
pattern very *
rewrite ?
(text_vectorization/StaticRegexReplace_60StaticRegexReplace1text_vectorization/StaticRegexReplace_59:output:0*#
_output_shapes
:?????????*
pattern	 weren't *
rewrite ?
(text_vectorization/StaticRegexReplace_61StaticRegexReplace1text_vectorization/StaticRegexReplace_60:output:0*#
_output_shapes
:?????????*
pattern been *
rewrite ?
(text_vectorization/StaticRegexReplace_62StaticRegexReplace1text_vectorization/StaticRegexReplace_61:output:0*#
_output_shapes
:?????????*
pattern will *
rewrite ?
(text_vectorization/StaticRegexReplace_63StaticRegexReplace1text_vectorization/StaticRegexReplace_62:output:0*#
_output_shapes
:?????????*
pattern now *
rewrite ?
(text_vectorization/StaticRegexReplace_64StaticRegexReplace1text_vectorization/StaticRegexReplace_63:output:0*#
_output_shapes
:?????????*
pattern they *
rewrite ?
(text_vectorization/StaticRegexReplace_65StaticRegexReplace1text_vectorization/StaticRegexReplace_64:output:0*#
_output_shapes
:?????????*
pattern when *
rewrite ?
(text_vectorization/StaticRegexReplace_66StaticRegexReplace1text_vectorization/StaticRegexReplace_65:output:0*#
_output_shapes
:?????????*
pattern was *
rewrite ?
(text_vectorization/StaticRegexReplace_67StaticRegexReplace1text_vectorization/StaticRegexReplace_66:output:0*#
_output_shapes
:?????????*
pattern shouldn't *
rewrite ?
(text_vectorization/StaticRegexReplace_68StaticRegexReplace1text_vectorization/StaticRegexReplace_67:output:0*#
_output_shapes
:?????????*
pattern	 herself *
rewrite ?
(text_vectorization/StaticRegexReplace_69StaticRegexReplace1text_vectorization/StaticRegexReplace_68:output:0*#
_output_shapes
:?????????*
pattern	 above *
rewrite ?
(text_vectorization/StaticRegexReplace_70StaticRegexReplace1text_vectorization/StaticRegexReplace_69:output:0*#
_output_shapes
:?????????*
pattern why *
rewrite ?
(text_vectorization/StaticRegexReplace_71StaticRegexReplace1text_vectorization/StaticRegexReplace_70:output:0*#
_output_shapes
:?????????*
pattern her *
rewrite ?
(text_vectorization/StaticRegexReplace_72StaticRegexReplace1text_vectorization/StaticRegexReplace_71:output:0*#
_output_shapes
:?????????*
pattern same *
rewrite ?
(text_vectorization/StaticRegexReplace_73StaticRegexReplace1text_vectorization/StaticRegexReplace_72:output:0*#
_output_shapes
:?????????*
pattern
 having *
rewrite ?
(text_vectorization/StaticRegexReplace_74StaticRegexReplace1text_vectorization/StaticRegexReplace_73:output:0*#
_output_shapes
:?????????*
pattern	 yours *
rewrite ?
(text_vectorization/StaticRegexReplace_75StaticRegexReplace1text_vectorization/StaticRegexReplace_74:output:0*#
_output_shapes
:?????????*
pattern can *
rewrite ?
(text_vectorization/StaticRegexReplace_76StaticRegexReplace1text_vectorization/StaticRegexReplace_75:output:0*#
_output_shapes
:?????????*
pattern
 wouldn't *
rewrite ?
(text_vectorization/StaticRegexReplace_77StaticRegexReplace1text_vectorization/StaticRegexReplace_76:output:0*#
_output_shapes
:?????????*
pattern	 again *
rewrite ?
(text_vectorization/StaticRegexReplace_78StaticRegexReplace1text_vectorization/StaticRegexReplace_77:output:0*#
_output_shapes
:?????????*
pattern do *
rewrite ?
(text_vectorization/StaticRegexReplace_79StaticRegexReplace1text_vectorization/StaticRegexReplace_78:output:0*#
_output_shapes
:?????????*
pattern shan *
rewrite ?
(text_vectorization/StaticRegexReplace_80StaticRegexReplace1text_vectorization/StaticRegexReplace_79:output:0*#
_output_shapes
:?????????*
pattern	 she's *
rewrite ?
(text_vectorization/StaticRegexReplace_81StaticRegexReplace1text_vectorization/StaticRegexReplace_80:output:0*#
_output_shapes
:?????????*
pattern of *
rewrite ?
(text_vectorization/StaticRegexReplace_82StaticRegexReplace1text_vectorization/StaticRegexReplace_81:output:0*#
_output_shapes
:?????????*
pattern	 against *
rewrite ?
(text_vectorization/StaticRegexReplace_83StaticRegexReplace1text_vectorization/StaticRegexReplace_82:output:0*#
_output_shapes
:?????????*
pattern most *
rewrite ?
(text_vectorization/StaticRegexReplace_84StaticRegexReplace1text_vectorization/StaticRegexReplace_83:output:0*#
_output_shapes
:?????????*
pattern	 isn't *
rewrite ?
(text_vectorization/StaticRegexReplace_85StaticRegexReplace1text_vectorization/StaticRegexReplace_84:output:0*#
_output_shapes
:?????????*
pattern	 until *
rewrite ?
(text_vectorization/StaticRegexReplace_86StaticRegexReplace1text_vectorization/StaticRegexReplace_85:output:0*#
_output_shapes
:?????????*
pattern it *
rewrite ?
(text_vectorization/StaticRegexReplace_87StaticRegexReplace1text_vectorization/StaticRegexReplace_86:output:0*#
_output_shapes
:?????????*
pattern	 below *
rewrite ?
(text_vectorization/StaticRegexReplace_88StaticRegexReplace1text_vectorization/StaticRegexReplace_87:output:0*#
_output_shapes
:?????????*
pattern	 mustn't *
rewrite ?
(text_vectorization/StaticRegexReplace_89StaticRegexReplace1text_vectorization/StaticRegexReplace_88:output:0*#
_output_shapes
:?????????*
pattern by *
rewrite ?
(text_vectorization/StaticRegexReplace_90StaticRegexReplace1text_vectorization/StaticRegexReplace_89:output:0*#
_output_shapes
:?????????*
pattern didn *
rewrite ?
(text_vectorization/StaticRegexReplace_91StaticRegexReplace1text_vectorization/StaticRegexReplace_90:output:0*#
_output_shapes
:?????????*
pattern
 shan't *
rewrite ?
(text_vectorization/StaticRegexReplace_92StaticRegexReplace1text_vectorization/StaticRegexReplace_91:output:0*#
_output_shapes
:?????????*
pattern who *
rewrite ?
(text_vectorization/StaticRegexReplace_93StaticRegexReplace1text_vectorization/StaticRegexReplace_92:output:0*#
_output_shapes
:?????????*
pattern both *
rewrite ?
(text_vectorization/StaticRegexReplace_94StaticRegexReplace1text_vectorization/StaticRegexReplace_93:output:0*#
_output_shapes
:?????????*
pattern re *
rewrite ?
(text_vectorization/StaticRegexReplace_95StaticRegexReplace1text_vectorization/StaticRegexReplace_94:output:0*#
_output_shapes
:?????????*
pattern
 wouldn *
rewrite ?
(text_vectorization/StaticRegexReplace_96StaticRegexReplace1text_vectorization/StaticRegexReplace_95:output:0*#
_output_shapes
:?????????*
pattern his *
rewrite ?
(text_vectorization/StaticRegexReplace_97StaticRegexReplace1text_vectorization/StaticRegexReplace_96:output:0*#
_output_shapes
:?????????*
pattern ours *
rewrite ?
(text_vectorization/StaticRegexReplace_98StaticRegexReplace1text_vectorization/StaticRegexReplace_97:output:0*#
_output_shapes
:?????????*
pattern
 itself *
rewrite ?
(text_vectorization/StaticRegexReplace_99StaticRegexReplace1text_vectorization/StaticRegexReplace_98:output:0*#
_output_shapes
:?????????*
pattern don *
rewrite ?
)text_vectorization/StaticRegexReplace_100StaticRegexReplace1text_vectorization/StaticRegexReplace_99:output:0*#
_output_shapes
:?????????*
pattern	 about *
rewrite ?
)text_vectorization/StaticRegexReplace_101StaticRegexReplace2text_vectorization/StaticRegexReplace_100:output:0*#
_output_shapes
:?????????*
pattern o *
rewrite ?
)text_vectorization/StaticRegexReplace_102StaticRegexReplace2text_vectorization/StaticRegexReplace_101:output:0*#
_output_shapes
:?????????*
pattern
 during *
rewrite ?
)text_vectorization/StaticRegexReplace_103StaticRegexReplace2text_vectorization/StaticRegexReplace_102:output:0*#
_output_shapes
:?????????*
pattern whom *
rewrite ?
)text_vectorization/StaticRegexReplace_104StaticRegexReplace2text_vectorization/StaticRegexReplace_103:output:0*#
_output_shapes
:?????????*
pattern
 mightn't *
rewrite ?
)text_vectorization/StaticRegexReplace_105StaticRegexReplace2text_vectorization/StaticRegexReplace_104:output:0*#
_output_shapes
:?????????*
pattern
 didn't *
rewrite ?
)text_vectorization/StaticRegexReplace_106StaticRegexReplace2text_vectorization/StaticRegexReplace_105:output:0*#
_output_shapes
:?????????*
pattern themselves *
rewrite ?
)text_vectorization/StaticRegexReplace_107StaticRegexReplace2text_vectorization/StaticRegexReplace_106:output:0*#
_output_shapes
:?????????*
pattern with *
rewrite ?
)text_vectorization/StaticRegexReplace_108StaticRegexReplace2text_vectorization/StaticRegexReplace_107:output:0*#
_output_shapes
:?????????*
pattern
 theirs *
rewrite ?
)text_vectorization/StaticRegexReplace_109StaticRegexReplace2text_vectorization/StaticRegexReplace_108:output:0*#
_output_shapes
:?????????*
pattern	 further *
rewrite ?
)text_vectorization/StaticRegexReplace_110StaticRegexReplace2text_vectorization/StaticRegexReplace_109:output:0*#
_output_shapes
:?????????*
pattern be *
rewrite ?
)text_vectorization/StaticRegexReplace_111StaticRegexReplace2text_vectorization/StaticRegexReplace_110:output:0*#
_output_shapes
:?????????*
pattern	 weren *
rewrite ?
)text_vectorization/StaticRegexReplace_112StaticRegexReplace2text_vectorization/StaticRegexReplace_111:output:0*#
_output_shapes
:?????????*
pattern own *
rewrite ?
)text_vectorization/StaticRegexReplace_113StaticRegexReplace2text_vectorization/StaticRegexReplace_112:output:0*#
_output_shapes
:?????????*
pattern into *
rewrite ?
)text_vectorization/StaticRegexReplace_114StaticRegexReplace2text_vectorization/StaticRegexReplace_113:output:0*#
_output_shapes
:?????????*
pattern t *
rewrite ?
)text_vectorization/StaticRegexReplace_115StaticRegexReplace2text_vectorization/StaticRegexReplace_114:output:0*#
_output_shapes
:?????????*
pattern	 haven *
rewrite ?
)text_vectorization/StaticRegexReplace_116StaticRegexReplace2text_vectorization/StaticRegexReplace_115:output:0*#
_output_shapes
:?????????*
pattern	 there *
rewrite ?
)text_vectorization/StaticRegexReplace_117StaticRegexReplace2text_vectorization/StaticRegexReplace_116:output:0*#
_output_shapes
:?????????*
pattern yourselves *
rewrite ?
)text_vectorization/StaticRegexReplace_118StaticRegexReplace2text_vectorization/StaticRegexReplace_117:output:0*#
_output_shapes
:?????????*
pattern
 aren't *
rewrite ?
)text_vectorization/StaticRegexReplace_119StaticRegexReplace2text_vectorization/StaticRegexReplace_118:output:0*#
_output_shapes
:?????????*
pattern
 you'll *
rewrite ?
)text_vectorization/StaticRegexReplace_120StaticRegexReplace2text_vectorization/StaticRegexReplace_119:output:0*#
_output_shapes
:?????????*
pattern how *
rewrite ?
)text_vectorization/StaticRegexReplace_121StaticRegexReplace2text_vectorization/StaticRegexReplace_120:output:0*#
_output_shapes
:?????????*
pattern ourselves *
rewrite ?
)text_vectorization/StaticRegexReplace_122StaticRegexReplace2text_vectorization/StaticRegexReplace_121:output:0*#
_output_shapes
:?????????*
pattern an *
rewrite ?
)text_vectorization/StaticRegexReplace_123StaticRegexReplace2text_vectorization/StaticRegexReplace_122:output:0*#
_output_shapes
:?????????*
pattern	 don't *
rewrite ?
)text_vectorization/StaticRegexReplace_124StaticRegexReplace2text_vectorization/StaticRegexReplace_123:output:0*#
_output_shapes
:?????????*
pattern	 doing *
rewrite ?
)text_vectorization/StaticRegexReplace_125StaticRegexReplace2text_vectorization/StaticRegexReplace_124:output:0*#
_output_shapes
:?????????*
pattern more *
rewrite ?
)text_vectorization/StaticRegexReplace_126StaticRegexReplace2text_vectorization/StaticRegexReplace_125:output:0*#
_output_shapes
:?????????*
pattern each *
rewrite ?
)text_vectorization/StaticRegexReplace_127StaticRegexReplace2text_vectorization/StaticRegexReplace_126:output:0*#
_output_shapes
:?????????*
pattern we *
rewrite ?
)text_vectorization/StaticRegexReplace_128StaticRegexReplace2text_vectorization/StaticRegexReplace_127:output:0*#
_output_shapes
:?????????*
pattern	 these *
rewrite ?
)text_vectorization/StaticRegexReplace_129StaticRegexReplace2text_vectorization/StaticRegexReplace_128:output:0*#
_output_shapes
:?????????*
pattern over *
rewrite ?
)text_vectorization/StaticRegexReplace_130StaticRegexReplace2text_vectorization/StaticRegexReplace_129:output:0*#
_output_shapes
:?????????*
pattern i *
rewrite ?
)text_vectorization/StaticRegexReplace_131StaticRegexReplace2text_vectorization/StaticRegexReplace_130:output:0*#
_output_shapes
:?????????*
pattern nor *
rewrite ?
)text_vectorization/StaticRegexReplace_132StaticRegexReplace2text_vectorization/StaticRegexReplace_131:output:0*#
_output_shapes
:?????????*
pattern	 needn't *
rewrite ?
)text_vectorization/StaticRegexReplace_133StaticRegexReplace2text_vectorization/StaticRegexReplace_132:output:0*#
_output_shapes
:?????????*
pattern ll *
rewrite ?
)text_vectorization/StaticRegexReplace_134StaticRegexReplace2text_vectorization/StaticRegexReplace_133:output:0*#
_output_shapes
:?????????*
pattern	 between *
rewrite ?
)text_vectorization/StaticRegexReplace_135StaticRegexReplace2text_vectorization/StaticRegexReplace_134:output:0*#
_output_shapes
:?????????*
pattern should've *
rewrite ?
)text_vectorization/StaticRegexReplace_136StaticRegexReplace2text_vectorization/StaticRegexReplace_135:output:0*#
_output_shapes
:?????????*
pattern
 hadn't *
rewrite ?
)text_vectorization/StaticRegexReplace_137StaticRegexReplace2text_vectorization/StaticRegexReplace_136:output:0*#
_output_shapes
:?????????*
pattern hasn *
rewrite ?
)text_vectorization/StaticRegexReplace_138StaticRegexReplace2text_vectorization/StaticRegexReplace_137:output:0*#
_output_shapes
:?????????*
pattern were *
rewrite ?
)text_vectorization/StaticRegexReplace_139StaticRegexReplace2text_vectorization/StaticRegexReplace_138:output:0*#
_output_shapes
:?????????*
pattern has *
rewrite ?
)text_vectorization/StaticRegexReplace_140StaticRegexReplace2text_vectorization/StaticRegexReplace_139:output:0*#
_output_shapes
:?????????*
pattern only *
rewrite ?
)text_vectorization/StaticRegexReplace_141StaticRegexReplace2text_vectorization/StaticRegexReplace_140:output:0*#
_output_shapes
:?????????*
pattern she *
rewrite ?
)text_vectorization/StaticRegexReplace_142StaticRegexReplace2text_vectorization/StaticRegexReplace_141:output:0*#
_output_shapes
:?????????*
pattern	 needn *
rewrite ?
)text_vectorization/StaticRegexReplace_143StaticRegexReplace2text_vectorization/StaticRegexReplace_142:output:0*#
_output_shapes
:?????????*
pattern	 other *
rewrite ?
)text_vectorization/StaticRegexReplace_144StaticRegexReplace2text_vectorization/StaticRegexReplace_143:output:0*#
_output_shapes
:?????????*
pattern
 hasn't *
rewrite ?
)text_vectorization/StaticRegexReplace_145StaticRegexReplace2text_vectorization/StaticRegexReplace_144:output:0*#
_output_shapes
:?????????*
pattern a *
rewrite ?
)text_vectorization/StaticRegexReplace_146StaticRegexReplace2text_vectorization/StaticRegexReplace_145:output:0*#
_output_shapes
:?????????*
pattern	 shouldn *
rewrite ?
)text_vectorization/StaticRegexReplace_147StaticRegexReplace2text_vectorization/StaticRegexReplace_146:output:0*#
_output_shapes
:?????????*
pattern and *
rewrite ?
)text_vectorization/StaticRegexReplace_148StaticRegexReplace2text_vectorization/StaticRegexReplace_147:output:0*#
_output_shapes
:?????????*
pattern	 those *
rewrite ?
)text_vectorization/StaticRegexReplace_149StaticRegexReplace2text_vectorization/StaticRegexReplace_148:output:0*#
_output_shapes
:?????????*
pattern	 being *
rewrite ?
)text_vectorization/StaticRegexReplace_150StaticRegexReplace2text_vectorization/StaticRegexReplace_149:output:0*#
_output_shapes
:?????????*
pattern such *
rewrite ?
)text_vectorization/StaticRegexReplace_151StaticRegexReplace2text_vectorization/StaticRegexReplace_150:output:0*#
_output_shapes
:?????????*
pattern as *
rewrite ?
)text_vectorization/StaticRegexReplace_152StaticRegexReplace2text_vectorization/StaticRegexReplace_151:output:0*#
_output_shapes
:?????????*
pattern ve *
rewrite ?
)text_vectorization/StaticRegexReplace_153StaticRegexReplace2text_vectorization/StaticRegexReplace_152:output:0*#
_output_shapes
:?????????*
pattern hers *
rewrite ?
)text_vectorization/StaticRegexReplace_154StaticRegexReplace2text_vectorization/StaticRegexReplace_153:output:0*#
_output_shapes
:?????????*
pattern s *
rewrite ?
)text_vectorization/StaticRegexReplace_155StaticRegexReplace2text_vectorization/StaticRegexReplace_154:output:0*#
_output_shapes
:?????????*
pattern	 their *
rewrite ?
)text_vectorization/StaticRegexReplace_156StaticRegexReplace2text_vectorization/StaticRegexReplace_155:output:0*#
_output_shapes
:?????????*
pattern	 haven't *
rewrite ?
)text_vectorization/StaticRegexReplace_157StaticRegexReplace2text_vectorization/StaticRegexReplace_156:output:0*#
_output_shapes
:?????????*
pattern for *
rewrite ?
)text_vectorization/StaticRegexReplace_158StaticRegexReplace2text_vectorization/StaticRegexReplace_157:output:0*#
_output_shapes
:?????????*
pattern if *
rewrite ?
)text_vectorization/StaticRegexReplace_159StaticRegexReplace2text_vectorization/StaticRegexReplace_158:output:0*#
_output_shapes
:?????????*
pattern that *
rewrite ?
)text_vectorization/StaticRegexReplace_160StaticRegexReplace2text_vectorization/StaticRegexReplace_159:output:0*#
_output_shapes
:?????????*
pattern isn *
rewrite ?
)text_vectorization/StaticRegexReplace_161StaticRegexReplace2text_vectorization/StaticRegexReplace_160:output:0*#
_output_shapes
:?????????*
pattern him *
rewrite ?
)text_vectorization/StaticRegexReplace_162StaticRegexReplace2text_vectorization/StaticRegexReplace_161:output:0*#
_output_shapes
:?????????*
pattern wasn *
rewrite ?
)text_vectorization/StaticRegexReplace_163StaticRegexReplace2text_vectorization/StaticRegexReplace_162:output:0*#
_output_shapes
:?????????*
pattern any *
rewrite ?
)text_vectorization/StaticRegexReplace_164StaticRegexReplace2text_vectorization/StaticRegexReplace_163:output:0*#
_output_shapes
:?????????*
pattern have *
rewrite ?
)text_vectorization/StaticRegexReplace_165StaticRegexReplace2text_vectorization/StaticRegexReplace_164:output:0*#
_output_shapes
:?????????*
pattern	 under *
rewrite ?
)text_vectorization/StaticRegexReplace_166StaticRegexReplace2text_vectorization/StaticRegexReplace_165:output:0*#
_output_shapes
:?????????*
pattern	 that'll *
rewrite ?
)text_vectorization/StaticRegexReplace_167StaticRegexReplace2text_vectorization/StaticRegexReplace_166:output:0*#
_output_shapes
:?????????*
pattern or *
rewrite ?
)text_vectorization/StaticRegexReplace_168StaticRegexReplace2text_vectorization/StaticRegexReplace_167:output:0*#
_output_shapes
:?????????*
pattern no *
rewrite ?
)text_vectorization/StaticRegexReplace_169StaticRegexReplace2text_vectorization/StaticRegexReplace_168:output:0*#
_output_shapes
:?????????*
pattern he *
rewrite ?
)text_vectorization/StaticRegexReplace_170StaticRegexReplace2text_vectorization/StaticRegexReplace_169:output:0*#
_output_shapes
:?????????*
pattern
 you're *
rewrite ?
)text_vectorization/StaticRegexReplace_171StaticRegexReplace2text_vectorization/StaticRegexReplace_170:output:0*#
_output_shapes
:?????????*
pattern this *
rewrite ?
)text_vectorization/StaticRegexReplace_172StaticRegexReplace2text_vectorization/StaticRegexReplace_171:output:0*#
_output_shapes
:?????????*
pattern	 doesn *
rewrite ?
)text_vectorization/StaticRegexReplace_173StaticRegexReplace2text_vectorization/StaticRegexReplace_172:output:0*#
_output_shapes
:?????????*
pattern	 you'd *
rewrite ?
)text_vectorization/StaticRegexReplace_174StaticRegexReplace2text_vectorization/StaticRegexReplace_173:output:0*#
_output_shapes
:?????????*
pattern up *
rewrite ?
)text_vectorization/StaticRegexReplace_175StaticRegexReplace2text_vectorization/StaticRegexReplace_174:output:0*#
_output_shapes
:?????????*
pattern
 you've *
rewrite ?
)text_vectorization/StaticRegexReplace_176StaticRegexReplace2text_vectorization/StaticRegexReplace_175:output:0*#
_output_shapes
:?????????*
pattern your *
rewrite ?
)text_vectorization/StaticRegexReplace_177StaticRegexReplace2text_vectorization/StaticRegexReplace_176:output:0*#
_output_shapes
:?????????*
pattern at *
rewrite ?
)text_vectorization/StaticRegexReplace_178StaticRegexReplace2text_vectorization/StaticRegexReplace_177:output:0*#
_output_shapes
:?????????*
pattern few *
rewrite ?
)text_vectorization/StaticRegexReplace_179StaticRegexReplace2text_vectorization/StaticRegexReplace_178:output:0*#
_output_shapes
:?????????*
pattern its *
rewrite ?
)text_vectorization/StaticRegexReplace_180StaticRegexReplace2text_vectorization/StaticRegexReplace_179:output:0*#
_output_shapes
:?????????*
pattern y *
rewrite ?
)text_vectorization/StaticRegexReplace_181StaticRegexReplace2text_vectorization/StaticRegexReplace_180:output:0*#
_output_shapes
:?????????*
pattern down *
rewrite ?
)text_vectorization/StaticRegexReplace_182StaticRegexReplace2text_vectorization/StaticRegexReplace_181:output:0*#
_output_shapes
:?????????*A
pattern64[!"\#\$%\&'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}\~]*
rewrite e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV22text_vectorization/StaticRegexReplace_182:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tGtext_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????d       ?
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????d*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
!embedding/StatefulPartitionedCallStatefulPartitionedCall?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0embedding_78761*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_77846?
(global_average_pooling1d/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_77598?
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_78765dense_78767*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_77862?
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_77943?
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_78771dense_1_78773*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_77885w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^embedding/StatefulPartitionedCall?^text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2?
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:] Y
#
_output_shapes
:?????????
2
_user_specified_nametext_vectorization_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
*
__inference_<lambda>_79764
identityJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
T
8__inference_global_average_pooling1d_layer_call_fn_79619

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_77598i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
__inference_<lambda>_797597
3key_value_init3446_lookuptableimportv2_table_handle/
+key_value_init3446_lookuptableimportv2_keys1
-key_value_init3446_lookuptableimportv2_values	
identity??&key_value_init3446/LookupTableImportV2?
&key_value_init3446/LookupTableImportV2LookupTableImportV23key_value_init3446_lookuptableimportv2_table_handle+key_value_init3446_lookuptableimportv2_keys-key_value_init3446_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: o
NoOpNoOp'^key_value_init3446/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?N:?N2P
&key_value_init3446/LookupTableImportV2&key_value_init3446/LookupTableImportV2:!

_output_shapes	
:?N:!

_output_shapes	
:?N
?
o
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_79625

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
??
?
E__inference_sequential_layer_call_and_return_conditional_losses_77892

inputsO
Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	"
embedding_77847:	?N 
dense_77863:  
dense_77865: 
dense_1_77886: 
dense_1_77888:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?!embedding/StatefulPartitionedCall?>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2Z
text_vectorization/StringLowerStringLowerinputs*#
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*#
_output_shapes
:?????????*
pattern<br />*
rewrite ?
'text_vectorization/StaticRegexReplace_1StaticRegexReplace.text_vectorization/StaticRegexReplace:output:0*#
_output_shapes
:?????????*+
pattern \d+(?:\.\d*)?(?:[eE][+-]?\d+)?*
rewrite ?
'text_vectorization/StaticRegexReplace_2StaticRegexReplace0text_vectorization/StaticRegexReplace_1:output:0*#
_output_shapes
:?????????*
pattern@([A-Za-z0-9_]+)*
rewrite ?
'text_vectorization/StaticRegexReplace_3StaticRegexReplace0text_vectorization/StaticRegexReplace_2:output:0*#
_output_shapes
:?????????*
pattern	 which *
rewrite ?
'text_vectorization/StaticRegexReplace_4StaticRegexReplace0text_vectorization/StaticRegexReplace_3:output:0*#
_output_shapes
:?????????*
pattern
 couldn *
rewrite ?
'text_vectorization/StaticRegexReplace_5StaticRegexReplace0text_vectorization/StaticRegexReplace_4:output:0*#
_output_shapes
:?????????*
pattern once *
rewrite ?
'text_vectorization/StaticRegexReplace_6StaticRegexReplace0text_vectorization/StaticRegexReplace_5:output:0*#
_output_shapes
:?????????*
pattern is *
rewrite ?
'text_vectorization/StaticRegexReplace_7StaticRegexReplace0text_vectorization/StaticRegexReplace_6:output:0*#
_output_shapes
:?????????*
pattern on *
rewrite ?
'text_vectorization/StaticRegexReplace_8StaticRegexReplace0text_vectorization/StaticRegexReplace_7:output:0*#
_output_shapes
:?????????*
pattern some *
rewrite ?
'text_vectorization/StaticRegexReplace_9StaticRegexReplace0text_vectorization/StaticRegexReplace_8:output:0*#
_output_shapes
:?????????*
pattern not *
rewrite ?
(text_vectorization/StaticRegexReplace_10StaticRegexReplace0text_vectorization/StaticRegexReplace_9:output:0*#
_output_shapes
:?????????*
pattern won *
rewrite ?
(text_vectorization/StaticRegexReplace_11StaticRegexReplace1text_vectorization/StaticRegexReplace_10:output:0*#
_output_shapes
:?????????*
pattern	 while *
rewrite ?
(text_vectorization/StaticRegexReplace_12StaticRegexReplace1text_vectorization/StaticRegexReplace_11:output:0*#
_output_shapes
:?????????*
pattern them *
rewrite ?
(text_vectorization/StaticRegexReplace_13StaticRegexReplace1text_vectorization/StaticRegexReplace_12:output:0*#
_output_shapes
:?????????*
pattern am *
rewrite ?
(text_vectorization/StaticRegexReplace_14StaticRegexReplace1text_vectorization/StaticRegexReplace_13:output:0*#
_output_shapes
:?????????*
pattern	 where *
rewrite ?
(text_vectorization/StaticRegexReplace_15StaticRegexReplace1text_vectorization/StaticRegexReplace_14:output:0*#
_output_shapes
:?????????*
pattern my *
rewrite ?
(text_vectorization/StaticRegexReplace_16StaticRegexReplace1text_vectorization/StaticRegexReplace_15:output:0*#
_output_shapes
:?????????*
pattern me *
rewrite ?
(text_vectorization/StaticRegexReplace_17StaticRegexReplace1text_vectorization/StaticRegexReplace_16:output:0*#
_output_shapes
:?????????*
pattern
 couldn't *
rewrite ?
(text_vectorization/StaticRegexReplace_18StaticRegexReplace1text_vectorization/StaticRegexReplace_17:output:0*#
_output_shapes
:?????????*
pattern all *
rewrite ?
(text_vectorization/StaticRegexReplace_19StaticRegexReplace1text_vectorization/StaticRegexReplace_18:output:0*#
_output_shapes
:?????????*
pattern it's *
rewrite ?
(text_vectorization/StaticRegexReplace_20StaticRegexReplace1text_vectorization/StaticRegexReplace_19:output:0*#
_output_shapes
:?????????*
pattern off *
rewrite ?
(text_vectorization/StaticRegexReplace_21StaticRegexReplace1text_vectorization/StaticRegexReplace_20:output:0*#
_output_shapes
:?????????*
pattern so *
rewrite ?
(text_vectorization/StaticRegexReplace_22StaticRegexReplace1text_vectorization/StaticRegexReplace_21:output:0*#
_output_shapes
:?????????*
pattern
 mightn *
rewrite ?
(text_vectorization/StaticRegexReplace_23StaticRegexReplace1text_vectorization/StaticRegexReplace_22:output:0*#
_output_shapes
:?????????*
pattern our *
rewrite ?
(text_vectorization/StaticRegexReplace_24StaticRegexReplace1text_vectorization/StaticRegexReplace_23:output:0*#
_output_shapes
:?????????*
pattern aren *
rewrite ?
(text_vectorization/StaticRegexReplace_25StaticRegexReplace1text_vectorization/StaticRegexReplace_24:output:0*#
_output_shapes
:?????????*
pattern	 won't *
rewrite ?
(text_vectorization/StaticRegexReplace_26StaticRegexReplace1text_vectorization/StaticRegexReplace_25:output:0*#
_output_shapes
:?????????*
pattern the *
rewrite ?
(text_vectorization/StaticRegexReplace_27StaticRegexReplace1text_vectorization/StaticRegexReplace_26:output:0*#
_output_shapes
:?????????*
pattern
 wasn't *
rewrite ?
(text_vectorization/StaticRegexReplace_28StaticRegexReplace1text_vectorization/StaticRegexReplace_27:output:0*#
_output_shapes
:?????????*
pattern just *
rewrite ?
(text_vectorization/StaticRegexReplace_29StaticRegexReplace1text_vectorization/StaticRegexReplace_28:output:0*#
_output_shapes
:?????????*
pattern
 myself *
rewrite ?
(text_vectorization/StaticRegexReplace_30StaticRegexReplace1text_vectorization/StaticRegexReplace_29:output:0*#
_output_shapes
:?????????*
pattern	 after *
rewrite ?
(text_vectorization/StaticRegexReplace_31StaticRegexReplace1text_vectorization/StaticRegexReplace_30:output:0*#
_output_shapes
:?????????*
pattern from *
rewrite ?
(text_vectorization/StaticRegexReplace_32StaticRegexReplace1text_vectorization/StaticRegexReplace_31:output:0*#
_output_shapes
:?????????*
pattern d *
rewrite ?
(text_vectorization/StaticRegexReplace_33StaticRegexReplace1text_vectorization/StaticRegexReplace_32:output:0*#
_output_shapes
:?????????*
pattern	 mustn *
rewrite ?
(text_vectorization/StaticRegexReplace_34StaticRegexReplace1text_vectorization/StaticRegexReplace_33:output:0*#
_output_shapes
:?????????*
pattern	 doesn't *
rewrite ?
(text_vectorization/StaticRegexReplace_35StaticRegexReplace1text_vectorization/StaticRegexReplace_34:output:0*#
_output_shapes
:?????????*
pattern did *
rewrite ?
(text_vectorization/StaticRegexReplace_36StaticRegexReplace1text_vectorization/StaticRegexReplace_35:output:0*#
_output_shapes
:?????????*
pattern what *
rewrite ?
(text_vectorization/StaticRegexReplace_37StaticRegexReplace1text_vectorization/StaticRegexReplace_36:output:0*#
_output_shapes
:?????????*
pattern in *
rewrite ?
(text_vectorization/StaticRegexReplace_38StaticRegexReplace1text_vectorization/StaticRegexReplace_37:output:0*#
_output_shapes
:?????????*
pattern out *
rewrite ?
(text_vectorization/StaticRegexReplace_39StaticRegexReplace1text_vectorization/StaticRegexReplace_38:output:0*#
_output_shapes
:?????????*
pattern than *
rewrite ?
(text_vectorization/StaticRegexReplace_40StaticRegexReplace1text_vectorization/StaticRegexReplace_39:output:0*#
_output_shapes
:?????????*
pattern to *
rewrite ?
(text_vectorization/StaticRegexReplace_41StaticRegexReplace1text_vectorization/StaticRegexReplace_40:output:0*#
_output_shapes
:?????????*
pattern	 because *
rewrite ?
(text_vectorization/StaticRegexReplace_42StaticRegexReplace1text_vectorization/StaticRegexReplace_41:output:0*#
_output_shapes
:?????????*
pattern too *
rewrite ?
(text_vectorization/StaticRegexReplace_43StaticRegexReplace1text_vectorization/StaticRegexReplace_42:output:0*#
_output_shapes
:?????????*
pattern here *
rewrite ?
(text_vectorization/StaticRegexReplace_44StaticRegexReplace1text_vectorization/StaticRegexReplace_43:output:0*#
_output_shapes
:?????????*
pattern ma *
rewrite ?
(text_vectorization/StaticRegexReplace_45StaticRegexReplace1text_vectorization/StaticRegexReplace_44:output:0*#
_output_shapes
:?????????*
pattern but *
rewrite ?
(text_vectorization/StaticRegexReplace_46StaticRegexReplace1text_vectorization/StaticRegexReplace_45:output:0*#
_output_shapes
:?????????*
pattern
 before *
rewrite ?
(text_vectorization/StaticRegexReplace_47StaticRegexReplace1text_vectorization/StaticRegexReplace_46:output:0*#
_output_shapes
:?????????*
pattern then *
rewrite ?
(text_vectorization/StaticRegexReplace_48StaticRegexReplace1text_vectorization/StaticRegexReplace_47:output:0*#
_output_shapes
:?????????*
pattern
 should *
rewrite ?
(text_vectorization/StaticRegexReplace_49StaticRegexReplace1text_vectorization/StaticRegexReplace_48:output:0*#
_output_shapes
:?????????*
pattern are *
rewrite ?
(text_vectorization/StaticRegexReplace_50StaticRegexReplace1text_vectorization/StaticRegexReplace_49:output:0*#
_output_shapes
:?????????*
pattern had *
rewrite ?
(text_vectorization/StaticRegexReplace_51StaticRegexReplace1text_vectorization/StaticRegexReplace_50:output:0*#
_output_shapes
:?????????*
pattern	 himself *
rewrite ?
(text_vectorization/StaticRegexReplace_52StaticRegexReplace1text_vectorization/StaticRegexReplace_51:output:0*#
_output_shapes
:?????????*
pattern you *
rewrite ?
(text_vectorization/StaticRegexReplace_53StaticRegexReplace1text_vectorization/StaticRegexReplace_52:output:0*#
_output_shapes
:?????????*
pattern
 yourself *
rewrite ?
(text_vectorization/StaticRegexReplace_54StaticRegexReplace1text_vectorization/StaticRegexReplace_53:output:0*#
_output_shapes
:?????????*
pattern	 through *
rewrite ?
(text_vectorization/StaticRegexReplace_55StaticRegexReplace1text_vectorization/StaticRegexReplace_54:output:0*#
_output_shapes
:?????????*
pattern hadn *
rewrite ?
(text_vectorization/StaticRegexReplace_56StaticRegexReplace1text_vectorization/StaticRegexReplace_55:output:0*#
_output_shapes
:?????????*
pattern does *
rewrite ?
(text_vectorization/StaticRegexReplace_57StaticRegexReplace1text_vectorization/StaticRegexReplace_56:output:0*#
_output_shapes
:?????????*
pattern m *
rewrite ?
(text_vectorization/StaticRegexReplace_58StaticRegexReplace1text_vectorization/StaticRegexReplace_57:output:0*#
_output_shapes
:?????????*
pattern ain *
rewrite ?
(text_vectorization/StaticRegexReplace_59StaticRegexReplace1text_vectorization/StaticRegexReplace_58:output:0*#
_output_shapes
:?????????*
pattern very *
rewrite ?
(text_vectorization/StaticRegexReplace_60StaticRegexReplace1text_vectorization/StaticRegexReplace_59:output:0*#
_output_shapes
:?????????*
pattern	 weren't *
rewrite ?
(text_vectorization/StaticRegexReplace_61StaticRegexReplace1text_vectorization/StaticRegexReplace_60:output:0*#
_output_shapes
:?????????*
pattern been *
rewrite ?
(text_vectorization/StaticRegexReplace_62StaticRegexReplace1text_vectorization/StaticRegexReplace_61:output:0*#
_output_shapes
:?????????*
pattern will *
rewrite ?
(text_vectorization/StaticRegexReplace_63StaticRegexReplace1text_vectorization/StaticRegexReplace_62:output:0*#
_output_shapes
:?????????*
pattern now *
rewrite ?
(text_vectorization/StaticRegexReplace_64StaticRegexReplace1text_vectorization/StaticRegexReplace_63:output:0*#
_output_shapes
:?????????*
pattern they *
rewrite ?
(text_vectorization/StaticRegexReplace_65StaticRegexReplace1text_vectorization/StaticRegexReplace_64:output:0*#
_output_shapes
:?????????*
pattern when *
rewrite ?
(text_vectorization/StaticRegexReplace_66StaticRegexReplace1text_vectorization/StaticRegexReplace_65:output:0*#
_output_shapes
:?????????*
pattern was *
rewrite ?
(text_vectorization/StaticRegexReplace_67StaticRegexReplace1text_vectorization/StaticRegexReplace_66:output:0*#
_output_shapes
:?????????*
pattern shouldn't *
rewrite ?
(text_vectorization/StaticRegexReplace_68StaticRegexReplace1text_vectorization/StaticRegexReplace_67:output:0*#
_output_shapes
:?????????*
pattern	 herself *
rewrite ?
(text_vectorization/StaticRegexReplace_69StaticRegexReplace1text_vectorization/StaticRegexReplace_68:output:0*#
_output_shapes
:?????????*
pattern	 above *
rewrite ?
(text_vectorization/StaticRegexReplace_70StaticRegexReplace1text_vectorization/StaticRegexReplace_69:output:0*#
_output_shapes
:?????????*
pattern why *
rewrite ?
(text_vectorization/StaticRegexReplace_71StaticRegexReplace1text_vectorization/StaticRegexReplace_70:output:0*#
_output_shapes
:?????????*
pattern her *
rewrite ?
(text_vectorization/StaticRegexReplace_72StaticRegexReplace1text_vectorization/StaticRegexReplace_71:output:0*#
_output_shapes
:?????????*
pattern same *
rewrite ?
(text_vectorization/StaticRegexReplace_73StaticRegexReplace1text_vectorization/StaticRegexReplace_72:output:0*#
_output_shapes
:?????????*
pattern
 having *
rewrite ?
(text_vectorization/StaticRegexReplace_74StaticRegexReplace1text_vectorization/StaticRegexReplace_73:output:0*#
_output_shapes
:?????????*
pattern	 yours *
rewrite ?
(text_vectorization/StaticRegexReplace_75StaticRegexReplace1text_vectorization/StaticRegexReplace_74:output:0*#
_output_shapes
:?????????*
pattern can *
rewrite ?
(text_vectorization/StaticRegexReplace_76StaticRegexReplace1text_vectorization/StaticRegexReplace_75:output:0*#
_output_shapes
:?????????*
pattern
 wouldn't *
rewrite ?
(text_vectorization/StaticRegexReplace_77StaticRegexReplace1text_vectorization/StaticRegexReplace_76:output:0*#
_output_shapes
:?????????*
pattern	 again *
rewrite ?
(text_vectorization/StaticRegexReplace_78StaticRegexReplace1text_vectorization/StaticRegexReplace_77:output:0*#
_output_shapes
:?????????*
pattern do *
rewrite ?
(text_vectorization/StaticRegexReplace_79StaticRegexReplace1text_vectorization/StaticRegexReplace_78:output:0*#
_output_shapes
:?????????*
pattern shan *
rewrite ?
(text_vectorization/StaticRegexReplace_80StaticRegexReplace1text_vectorization/StaticRegexReplace_79:output:0*#
_output_shapes
:?????????*
pattern	 she's *
rewrite ?
(text_vectorization/StaticRegexReplace_81StaticRegexReplace1text_vectorization/StaticRegexReplace_80:output:0*#
_output_shapes
:?????????*
pattern of *
rewrite ?
(text_vectorization/StaticRegexReplace_82StaticRegexReplace1text_vectorization/StaticRegexReplace_81:output:0*#
_output_shapes
:?????????*
pattern	 against *
rewrite ?
(text_vectorization/StaticRegexReplace_83StaticRegexReplace1text_vectorization/StaticRegexReplace_82:output:0*#
_output_shapes
:?????????*
pattern most *
rewrite ?
(text_vectorization/StaticRegexReplace_84StaticRegexReplace1text_vectorization/StaticRegexReplace_83:output:0*#
_output_shapes
:?????????*
pattern	 isn't *
rewrite ?
(text_vectorization/StaticRegexReplace_85StaticRegexReplace1text_vectorization/StaticRegexReplace_84:output:0*#
_output_shapes
:?????????*
pattern	 until *
rewrite ?
(text_vectorization/StaticRegexReplace_86StaticRegexReplace1text_vectorization/StaticRegexReplace_85:output:0*#
_output_shapes
:?????????*
pattern it *
rewrite ?
(text_vectorization/StaticRegexReplace_87StaticRegexReplace1text_vectorization/StaticRegexReplace_86:output:0*#
_output_shapes
:?????????*
pattern	 below *
rewrite ?
(text_vectorization/StaticRegexReplace_88StaticRegexReplace1text_vectorization/StaticRegexReplace_87:output:0*#
_output_shapes
:?????????*
pattern	 mustn't *
rewrite ?
(text_vectorization/StaticRegexReplace_89StaticRegexReplace1text_vectorization/StaticRegexReplace_88:output:0*#
_output_shapes
:?????????*
pattern by *
rewrite ?
(text_vectorization/StaticRegexReplace_90StaticRegexReplace1text_vectorization/StaticRegexReplace_89:output:0*#
_output_shapes
:?????????*
pattern didn *
rewrite ?
(text_vectorization/StaticRegexReplace_91StaticRegexReplace1text_vectorization/StaticRegexReplace_90:output:0*#
_output_shapes
:?????????*
pattern
 shan't *
rewrite ?
(text_vectorization/StaticRegexReplace_92StaticRegexReplace1text_vectorization/StaticRegexReplace_91:output:0*#
_output_shapes
:?????????*
pattern who *
rewrite ?
(text_vectorization/StaticRegexReplace_93StaticRegexReplace1text_vectorization/StaticRegexReplace_92:output:0*#
_output_shapes
:?????????*
pattern both *
rewrite ?
(text_vectorization/StaticRegexReplace_94StaticRegexReplace1text_vectorization/StaticRegexReplace_93:output:0*#
_output_shapes
:?????????*
pattern re *
rewrite ?
(text_vectorization/StaticRegexReplace_95StaticRegexReplace1text_vectorization/StaticRegexReplace_94:output:0*#
_output_shapes
:?????????*
pattern
 wouldn *
rewrite ?
(text_vectorization/StaticRegexReplace_96StaticRegexReplace1text_vectorization/StaticRegexReplace_95:output:0*#
_output_shapes
:?????????*
pattern his *
rewrite ?
(text_vectorization/StaticRegexReplace_97StaticRegexReplace1text_vectorization/StaticRegexReplace_96:output:0*#
_output_shapes
:?????????*
pattern ours *
rewrite ?
(text_vectorization/StaticRegexReplace_98StaticRegexReplace1text_vectorization/StaticRegexReplace_97:output:0*#
_output_shapes
:?????????*
pattern
 itself *
rewrite ?
(text_vectorization/StaticRegexReplace_99StaticRegexReplace1text_vectorization/StaticRegexReplace_98:output:0*#
_output_shapes
:?????????*
pattern don *
rewrite ?
)text_vectorization/StaticRegexReplace_100StaticRegexReplace1text_vectorization/StaticRegexReplace_99:output:0*#
_output_shapes
:?????????*
pattern	 about *
rewrite ?
)text_vectorization/StaticRegexReplace_101StaticRegexReplace2text_vectorization/StaticRegexReplace_100:output:0*#
_output_shapes
:?????????*
pattern o *
rewrite ?
)text_vectorization/StaticRegexReplace_102StaticRegexReplace2text_vectorization/StaticRegexReplace_101:output:0*#
_output_shapes
:?????????*
pattern
 during *
rewrite ?
)text_vectorization/StaticRegexReplace_103StaticRegexReplace2text_vectorization/StaticRegexReplace_102:output:0*#
_output_shapes
:?????????*
pattern whom *
rewrite ?
)text_vectorization/StaticRegexReplace_104StaticRegexReplace2text_vectorization/StaticRegexReplace_103:output:0*#
_output_shapes
:?????????*
pattern
 mightn't *
rewrite ?
)text_vectorization/StaticRegexReplace_105StaticRegexReplace2text_vectorization/StaticRegexReplace_104:output:0*#
_output_shapes
:?????????*
pattern
 didn't *
rewrite ?
)text_vectorization/StaticRegexReplace_106StaticRegexReplace2text_vectorization/StaticRegexReplace_105:output:0*#
_output_shapes
:?????????*
pattern themselves *
rewrite ?
)text_vectorization/StaticRegexReplace_107StaticRegexReplace2text_vectorization/StaticRegexReplace_106:output:0*#
_output_shapes
:?????????*
pattern with *
rewrite ?
)text_vectorization/StaticRegexReplace_108StaticRegexReplace2text_vectorization/StaticRegexReplace_107:output:0*#
_output_shapes
:?????????*
pattern
 theirs *
rewrite ?
)text_vectorization/StaticRegexReplace_109StaticRegexReplace2text_vectorization/StaticRegexReplace_108:output:0*#
_output_shapes
:?????????*
pattern	 further *
rewrite ?
)text_vectorization/StaticRegexReplace_110StaticRegexReplace2text_vectorization/StaticRegexReplace_109:output:0*#
_output_shapes
:?????????*
pattern be *
rewrite ?
)text_vectorization/StaticRegexReplace_111StaticRegexReplace2text_vectorization/StaticRegexReplace_110:output:0*#
_output_shapes
:?????????*
pattern	 weren *
rewrite ?
)text_vectorization/StaticRegexReplace_112StaticRegexReplace2text_vectorization/StaticRegexReplace_111:output:0*#
_output_shapes
:?????????*
pattern own *
rewrite ?
)text_vectorization/StaticRegexReplace_113StaticRegexReplace2text_vectorization/StaticRegexReplace_112:output:0*#
_output_shapes
:?????????*
pattern into *
rewrite ?
)text_vectorization/StaticRegexReplace_114StaticRegexReplace2text_vectorization/StaticRegexReplace_113:output:0*#
_output_shapes
:?????????*
pattern t *
rewrite ?
)text_vectorization/StaticRegexReplace_115StaticRegexReplace2text_vectorization/StaticRegexReplace_114:output:0*#
_output_shapes
:?????????*
pattern	 haven *
rewrite ?
)text_vectorization/StaticRegexReplace_116StaticRegexReplace2text_vectorization/StaticRegexReplace_115:output:0*#
_output_shapes
:?????????*
pattern	 there *
rewrite ?
)text_vectorization/StaticRegexReplace_117StaticRegexReplace2text_vectorization/StaticRegexReplace_116:output:0*#
_output_shapes
:?????????*
pattern yourselves *
rewrite ?
)text_vectorization/StaticRegexReplace_118StaticRegexReplace2text_vectorization/StaticRegexReplace_117:output:0*#
_output_shapes
:?????????*
pattern
 aren't *
rewrite ?
)text_vectorization/StaticRegexReplace_119StaticRegexReplace2text_vectorization/StaticRegexReplace_118:output:0*#
_output_shapes
:?????????*
pattern
 you'll *
rewrite ?
)text_vectorization/StaticRegexReplace_120StaticRegexReplace2text_vectorization/StaticRegexReplace_119:output:0*#
_output_shapes
:?????????*
pattern how *
rewrite ?
)text_vectorization/StaticRegexReplace_121StaticRegexReplace2text_vectorization/StaticRegexReplace_120:output:0*#
_output_shapes
:?????????*
pattern ourselves *
rewrite ?
)text_vectorization/StaticRegexReplace_122StaticRegexReplace2text_vectorization/StaticRegexReplace_121:output:0*#
_output_shapes
:?????????*
pattern an *
rewrite ?
)text_vectorization/StaticRegexReplace_123StaticRegexReplace2text_vectorization/StaticRegexReplace_122:output:0*#
_output_shapes
:?????????*
pattern	 don't *
rewrite ?
)text_vectorization/StaticRegexReplace_124StaticRegexReplace2text_vectorization/StaticRegexReplace_123:output:0*#
_output_shapes
:?????????*
pattern	 doing *
rewrite ?
)text_vectorization/StaticRegexReplace_125StaticRegexReplace2text_vectorization/StaticRegexReplace_124:output:0*#
_output_shapes
:?????????*
pattern more *
rewrite ?
)text_vectorization/StaticRegexReplace_126StaticRegexReplace2text_vectorization/StaticRegexReplace_125:output:0*#
_output_shapes
:?????????*
pattern each *
rewrite ?
)text_vectorization/StaticRegexReplace_127StaticRegexReplace2text_vectorization/StaticRegexReplace_126:output:0*#
_output_shapes
:?????????*
pattern we *
rewrite ?
)text_vectorization/StaticRegexReplace_128StaticRegexReplace2text_vectorization/StaticRegexReplace_127:output:0*#
_output_shapes
:?????????*
pattern	 these *
rewrite ?
)text_vectorization/StaticRegexReplace_129StaticRegexReplace2text_vectorization/StaticRegexReplace_128:output:0*#
_output_shapes
:?????????*
pattern over *
rewrite ?
)text_vectorization/StaticRegexReplace_130StaticRegexReplace2text_vectorization/StaticRegexReplace_129:output:0*#
_output_shapes
:?????????*
pattern i *
rewrite ?
)text_vectorization/StaticRegexReplace_131StaticRegexReplace2text_vectorization/StaticRegexReplace_130:output:0*#
_output_shapes
:?????????*
pattern nor *
rewrite ?
)text_vectorization/StaticRegexReplace_132StaticRegexReplace2text_vectorization/StaticRegexReplace_131:output:0*#
_output_shapes
:?????????*
pattern	 needn't *
rewrite ?
)text_vectorization/StaticRegexReplace_133StaticRegexReplace2text_vectorization/StaticRegexReplace_132:output:0*#
_output_shapes
:?????????*
pattern ll *
rewrite ?
)text_vectorization/StaticRegexReplace_134StaticRegexReplace2text_vectorization/StaticRegexReplace_133:output:0*#
_output_shapes
:?????????*
pattern	 between *
rewrite ?
)text_vectorization/StaticRegexReplace_135StaticRegexReplace2text_vectorization/StaticRegexReplace_134:output:0*#
_output_shapes
:?????????*
pattern should've *
rewrite ?
)text_vectorization/StaticRegexReplace_136StaticRegexReplace2text_vectorization/StaticRegexReplace_135:output:0*#
_output_shapes
:?????????*
pattern
 hadn't *
rewrite ?
)text_vectorization/StaticRegexReplace_137StaticRegexReplace2text_vectorization/StaticRegexReplace_136:output:0*#
_output_shapes
:?????????*
pattern hasn *
rewrite ?
)text_vectorization/StaticRegexReplace_138StaticRegexReplace2text_vectorization/StaticRegexReplace_137:output:0*#
_output_shapes
:?????????*
pattern were *
rewrite ?
)text_vectorization/StaticRegexReplace_139StaticRegexReplace2text_vectorization/StaticRegexReplace_138:output:0*#
_output_shapes
:?????????*
pattern has *
rewrite ?
)text_vectorization/StaticRegexReplace_140StaticRegexReplace2text_vectorization/StaticRegexReplace_139:output:0*#
_output_shapes
:?????????*
pattern only *
rewrite ?
)text_vectorization/StaticRegexReplace_141StaticRegexReplace2text_vectorization/StaticRegexReplace_140:output:0*#
_output_shapes
:?????????*
pattern she *
rewrite ?
)text_vectorization/StaticRegexReplace_142StaticRegexReplace2text_vectorization/StaticRegexReplace_141:output:0*#
_output_shapes
:?????????*
pattern	 needn *
rewrite ?
)text_vectorization/StaticRegexReplace_143StaticRegexReplace2text_vectorization/StaticRegexReplace_142:output:0*#
_output_shapes
:?????????*
pattern	 other *
rewrite ?
)text_vectorization/StaticRegexReplace_144StaticRegexReplace2text_vectorization/StaticRegexReplace_143:output:0*#
_output_shapes
:?????????*
pattern
 hasn't *
rewrite ?
)text_vectorization/StaticRegexReplace_145StaticRegexReplace2text_vectorization/StaticRegexReplace_144:output:0*#
_output_shapes
:?????????*
pattern a *
rewrite ?
)text_vectorization/StaticRegexReplace_146StaticRegexReplace2text_vectorization/StaticRegexReplace_145:output:0*#
_output_shapes
:?????????*
pattern	 shouldn *
rewrite ?
)text_vectorization/StaticRegexReplace_147StaticRegexReplace2text_vectorization/StaticRegexReplace_146:output:0*#
_output_shapes
:?????????*
pattern and *
rewrite ?
)text_vectorization/StaticRegexReplace_148StaticRegexReplace2text_vectorization/StaticRegexReplace_147:output:0*#
_output_shapes
:?????????*
pattern	 those *
rewrite ?
)text_vectorization/StaticRegexReplace_149StaticRegexReplace2text_vectorization/StaticRegexReplace_148:output:0*#
_output_shapes
:?????????*
pattern	 being *
rewrite ?
)text_vectorization/StaticRegexReplace_150StaticRegexReplace2text_vectorization/StaticRegexReplace_149:output:0*#
_output_shapes
:?????????*
pattern such *
rewrite ?
)text_vectorization/StaticRegexReplace_151StaticRegexReplace2text_vectorization/StaticRegexReplace_150:output:0*#
_output_shapes
:?????????*
pattern as *
rewrite ?
)text_vectorization/StaticRegexReplace_152StaticRegexReplace2text_vectorization/StaticRegexReplace_151:output:0*#
_output_shapes
:?????????*
pattern ve *
rewrite ?
)text_vectorization/StaticRegexReplace_153StaticRegexReplace2text_vectorization/StaticRegexReplace_152:output:0*#
_output_shapes
:?????????*
pattern hers *
rewrite ?
)text_vectorization/StaticRegexReplace_154StaticRegexReplace2text_vectorization/StaticRegexReplace_153:output:0*#
_output_shapes
:?????????*
pattern s *
rewrite ?
)text_vectorization/StaticRegexReplace_155StaticRegexReplace2text_vectorization/StaticRegexReplace_154:output:0*#
_output_shapes
:?????????*
pattern	 their *
rewrite ?
)text_vectorization/StaticRegexReplace_156StaticRegexReplace2text_vectorization/StaticRegexReplace_155:output:0*#
_output_shapes
:?????????*
pattern	 haven't *
rewrite ?
)text_vectorization/StaticRegexReplace_157StaticRegexReplace2text_vectorization/StaticRegexReplace_156:output:0*#
_output_shapes
:?????????*
pattern for *
rewrite ?
)text_vectorization/StaticRegexReplace_158StaticRegexReplace2text_vectorization/StaticRegexReplace_157:output:0*#
_output_shapes
:?????????*
pattern if *
rewrite ?
)text_vectorization/StaticRegexReplace_159StaticRegexReplace2text_vectorization/StaticRegexReplace_158:output:0*#
_output_shapes
:?????????*
pattern that *
rewrite ?
)text_vectorization/StaticRegexReplace_160StaticRegexReplace2text_vectorization/StaticRegexReplace_159:output:0*#
_output_shapes
:?????????*
pattern isn *
rewrite ?
)text_vectorization/StaticRegexReplace_161StaticRegexReplace2text_vectorization/StaticRegexReplace_160:output:0*#
_output_shapes
:?????????*
pattern him *
rewrite ?
)text_vectorization/StaticRegexReplace_162StaticRegexReplace2text_vectorization/StaticRegexReplace_161:output:0*#
_output_shapes
:?????????*
pattern wasn *
rewrite ?
)text_vectorization/StaticRegexReplace_163StaticRegexReplace2text_vectorization/StaticRegexReplace_162:output:0*#
_output_shapes
:?????????*
pattern any *
rewrite ?
)text_vectorization/StaticRegexReplace_164StaticRegexReplace2text_vectorization/StaticRegexReplace_163:output:0*#
_output_shapes
:?????????*
pattern have *
rewrite ?
)text_vectorization/StaticRegexReplace_165StaticRegexReplace2text_vectorization/StaticRegexReplace_164:output:0*#
_output_shapes
:?????????*
pattern	 under *
rewrite ?
)text_vectorization/StaticRegexReplace_166StaticRegexReplace2text_vectorization/StaticRegexReplace_165:output:0*#
_output_shapes
:?????????*
pattern	 that'll *
rewrite ?
)text_vectorization/StaticRegexReplace_167StaticRegexReplace2text_vectorization/StaticRegexReplace_166:output:0*#
_output_shapes
:?????????*
pattern or *
rewrite ?
)text_vectorization/StaticRegexReplace_168StaticRegexReplace2text_vectorization/StaticRegexReplace_167:output:0*#
_output_shapes
:?????????*
pattern no *
rewrite ?
)text_vectorization/StaticRegexReplace_169StaticRegexReplace2text_vectorization/StaticRegexReplace_168:output:0*#
_output_shapes
:?????????*
pattern he *
rewrite ?
)text_vectorization/StaticRegexReplace_170StaticRegexReplace2text_vectorization/StaticRegexReplace_169:output:0*#
_output_shapes
:?????????*
pattern
 you're *
rewrite ?
)text_vectorization/StaticRegexReplace_171StaticRegexReplace2text_vectorization/StaticRegexReplace_170:output:0*#
_output_shapes
:?????????*
pattern this *
rewrite ?
)text_vectorization/StaticRegexReplace_172StaticRegexReplace2text_vectorization/StaticRegexReplace_171:output:0*#
_output_shapes
:?????????*
pattern	 doesn *
rewrite ?
)text_vectorization/StaticRegexReplace_173StaticRegexReplace2text_vectorization/StaticRegexReplace_172:output:0*#
_output_shapes
:?????????*
pattern	 you'd *
rewrite ?
)text_vectorization/StaticRegexReplace_174StaticRegexReplace2text_vectorization/StaticRegexReplace_173:output:0*#
_output_shapes
:?????????*
pattern up *
rewrite ?
)text_vectorization/StaticRegexReplace_175StaticRegexReplace2text_vectorization/StaticRegexReplace_174:output:0*#
_output_shapes
:?????????*
pattern
 you've *
rewrite ?
)text_vectorization/StaticRegexReplace_176StaticRegexReplace2text_vectorization/StaticRegexReplace_175:output:0*#
_output_shapes
:?????????*
pattern your *
rewrite ?
)text_vectorization/StaticRegexReplace_177StaticRegexReplace2text_vectorization/StaticRegexReplace_176:output:0*#
_output_shapes
:?????????*
pattern at *
rewrite ?
)text_vectorization/StaticRegexReplace_178StaticRegexReplace2text_vectorization/StaticRegexReplace_177:output:0*#
_output_shapes
:?????????*
pattern few *
rewrite ?
)text_vectorization/StaticRegexReplace_179StaticRegexReplace2text_vectorization/StaticRegexReplace_178:output:0*#
_output_shapes
:?????????*
pattern its *
rewrite ?
)text_vectorization/StaticRegexReplace_180StaticRegexReplace2text_vectorization/StaticRegexReplace_179:output:0*#
_output_shapes
:?????????*
pattern y *
rewrite ?
)text_vectorization/StaticRegexReplace_181StaticRegexReplace2text_vectorization/StaticRegexReplace_180:output:0*#
_output_shapes
:?????????*
pattern down *
rewrite ?
)text_vectorization/StaticRegexReplace_182StaticRegexReplace2text_vectorization/StaticRegexReplace_181:output:0*#
_output_shapes
:?????????*A
pattern64[!"\#\$%\&'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}\~]*
rewrite e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV22text_vectorization/StaticRegexReplace_182:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tGtext_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????d       ?
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????d*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
!embedding/StatefulPartitionedCallStatefulPartitionedCall?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0embedding_77847*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_77846?
(global_average_pooling1d/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_77598?
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_77863dense_77865*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_77862?
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_77873?
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_77886dense_1_77888*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_77885w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^embedding/StatefulPartitionedCall?^text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2?
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?:
?
__inference__traced_save_79873
file_prefix3
/savev2_embedding_embeddings_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopJ
Fsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2L
Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1	&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop:
6savev2_adam_embedding_embeddings_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop:
6savev2_adam_embedding_embeddings_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop
savev2_const_6

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-valuesB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B ?

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0/savev2_embedding_embeddings_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopFsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop6savev2_adam_embedding_embeddings_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop6savev2_adam_embedding_embeddings_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableopsavev2_const_6"/device:CPU:0*
_output_shapes
 *)
dtypes
2		?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :	?N :  : : :: : : : : ::: : : : :	?N :  : : ::	?N :  : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?N :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?N :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::%!

_output_shapes
:	?N :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: 
?	
?
B__inference_dense_1_layer_call_and_return_conditional_losses_77885

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
a
B__inference_dropout_layer_call_and_return_conditional_losses_77943

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:????????? C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:????????? Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
__inference_save_fn_79743
checkpoint_keyP
Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2

identity_3

identity_4

identity_5	???MutableHashTable_lookup_table_export_values/LookupTableExportV2?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0*
Tvalues0	*
_output_shapes

::P
add/yConst*
_output_shapes
: *
dtype0*
valueB B
table-keysK
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: T
add_1/yConst*
_output_shapes
: *
dtype0*
valueB Btable-valuesO
add_1Addcheckpoint_keyadd_1/y:output:0*
T0*
_output_shapes
: E
IdentityIdentityadd:z:0^NoOp*
T0*
_output_shapes
: F
ConstConst*
_output_shapes
: *
dtype0*
valueB B N

Identity_1IdentityConst:output:0^NoOp*
T0*
_output_shapes
: ?

Identity_2IdentityFMutableHashTable_lookup_table_export_values/LookupTableExportV2:keys:0^NoOp*
T0*
_output_shapes
:I

Identity_3Identity	add_1:z:0^NoOp*
T0*
_output_shapes
: H
Const_1Const*
_output_shapes
: *
dtype0*
valueB B P

Identity_4IdentityConst_1:output:0^NoOp*
T0*
_output_shapes
: ?

Identity_5IdentityHMutableHashTable_lookup_table_export_values/LookupTableExportV2:values:0^NoOp*
T0	*
_output_shapes
:?
NoOpNoOp@^MutableHashTable_lookup_table_export_values/LookupTableExportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2?MutableHashTable_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key
??
?
__inference_adapt_step_79037
iterator9
5none_lookup_table_find_lookuptablefindv2_table_handle:
6none_lookup_table_find_lookuptablefindv2_default_value	??IteratorGetNext?(None_lookup_table_find/LookupTableFindV2?,None_lookup_table_insert/LookupTableInsertV2?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*#
_output_shapes
:?????????*"
output_shapes
:?????????*
output_types
2]
StringLowerStringLowerIteratorGetNext:components:0*#
_output_shapes
:??????????
StaticRegexReplaceStaticRegexReplaceStringLower:output:0*#
_output_shapes
:?????????*
pattern<br />*
rewrite ?
StaticRegexReplace_1StaticRegexReplaceStaticRegexReplace:output:0*#
_output_shapes
:?????????*+
pattern \d+(?:\.\d*)?(?:[eE][+-]?\d+)?*
rewrite ?
StaticRegexReplace_2StaticRegexReplaceStaticRegexReplace_1:output:0*#
_output_shapes
:?????????*
pattern@([A-Za-z0-9_]+)*
rewrite ?
StaticRegexReplace_3StaticRegexReplaceStaticRegexReplace_2:output:0*#
_output_shapes
:?????????*
pattern	 which *
rewrite ?
StaticRegexReplace_4StaticRegexReplaceStaticRegexReplace_3:output:0*#
_output_shapes
:?????????*
pattern
 couldn *
rewrite ?
StaticRegexReplace_5StaticRegexReplaceStaticRegexReplace_4:output:0*#
_output_shapes
:?????????*
pattern once *
rewrite ?
StaticRegexReplace_6StaticRegexReplaceStaticRegexReplace_5:output:0*#
_output_shapes
:?????????*
pattern is *
rewrite ?
StaticRegexReplace_7StaticRegexReplaceStaticRegexReplace_6:output:0*#
_output_shapes
:?????????*
pattern on *
rewrite ?
StaticRegexReplace_8StaticRegexReplaceStaticRegexReplace_7:output:0*#
_output_shapes
:?????????*
pattern some *
rewrite ?
StaticRegexReplace_9StaticRegexReplaceStaticRegexReplace_8:output:0*#
_output_shapes
:?????????*
pattern not *
rewrite ?
StaticRegexReplace_10StaticRegexReplaceStaticRegexReplace_9:output:0*#
_output_shapes
:?????????*
pattern won *
rewrite ?
StaticRegexReplace_11StaticRegexReplaceStaticRegexReplace_10:output:0*#
_output_shapes
:?????????*
pattern	 while *
rewrite ?
StaticRegexReplace_12StaticRegexReplaceStaticRegexReplace_11:output:0*#
_output_shapes
:?????????*
pattern them *
rewrite ?
StaticRegexReplace_13StaticRegexReplaceStaticRegexReplace_12:output:0*#
_output_shapes
:?????????*
pattern am *
rewrite ?
StaticRegexReplace_14StaticRegexReplaceStaticRegexReplace_13:output:0*#
_output_shapes
:?????????*
pattern	 where *
rewrite ?
StaticRegexReplace_15StaticRegexReplaceStaticRegexReplace_14:output:0*#
_output_shapes
:?????????*
pattern my *
rewrite ?
StaticRegexReplace_16StaticRegexReplaceStaticRegexReplace_15:output:0*#
_output_shapes
:?????????*
pattern me *
rewrite ?
StaticRegexReplace_17StaticRegexReplaceStaticRegexReplace_16:output:0*#
_output_shapes
:?????????*
pattern
 couldn't *
rewrite ?
StaticRegexReplace_18StaticRegexReplaceStaticRegexReplace_17:output:0*#
_output_shapes
:?????????*
pattern all *
rewrite ?
StaticRegexReplace_19StaticRegexReplaceStaticRegexReplace_18:output:0*#
_output_shapes
:?????????*
pattern it's *
rewrite ?
StaticRegexReplace_20StaticRegexReplaceStaticRegexReplace_19:output:0*#
_output_shapes
:?????????*
pattern off *
rewrite ?
StaticRegexReplace_21StaticRegexReplaceStaticRegexReplace_20:output:0*#
_output_shapes
:?????????*
pattern so *
rewrite ?
StaticRegexReplace_22StaticRegexReplaceStaticRegexReplace_21:output:0*#
_output_shapes
:?????????*
pattern
 mightn *
rewrite ?
StaticRegexReplace_23StaticRegexReplaceStaticRegexReplace_22:output:0*#
_output_shapes
:?????????*
pattern our *
rewrite ?
StaticRegexReplace_24StaticRegexReplaceStaticRegexReplace_23:output:0*#
_output_shapes
:?????????*
pattern aren *
rewrite ?
StaticRegexReplace_25StaticRegexReplaceStaticRegexReplace_24:output:0*#
_output_shapes
:?????????*
pattern	 won't *
rewrite ?
StaticRegexReplace_26StaticRegexReplaceStaticRegexReplace_25:output:0*#
_output_shapes
:?????????*
pattern the *
rewrite ?
StaticRegexReplace_27StaticRegexReplaceStaticRegexReplace_26:output:0*#
_output_shapes
:?????????*
pattern
 wasn't *
rewrite ?
StaticRegexReplace_28StaticRegexReplaceStaticRegexReplace_27:output:0*#
_output_shapes
:?????????*
pattern just *
rewrite ?
StaticRegexReplace_29StaticRegexReplaceStaticRegexReplace_28:output:0*#
_output_shapes
:?????????*
pattern
 myself *
rewrite ?
StaticRegexReplace_30StaticRegexReplaceStaticRegexReplace_29:output:0*#
_output_shapes
:?????????*
pattern	 after *
rewrite ?
StaticRegexReplace_31StaticRegexReplaceStaticRegexReplace_30:output:0*#
_output_shapes
:?????????*
pattern from *
rewrite ?
StaticRegexReplace_32StaticRegexReplaceStaticRegexReplace_31:output:0*#
_output_shapes
:?????????*
pattern d *
rewrite ?
StaticRegexReplace_33StaticRegexReplaceStaticRegexReplace_32:output:0*#
_output_shapes
:?????????*
pattern	 mustn *
rewrite ?
StaticRegexReplace_34StaticRegexReplaceStaticRegexReplace_33:output:0*#
_output_shapes
:?????????*
pattern	 doesn't *
rewrite ?
StaticRegexReplace_35StaticRegexReplaceStaticRegexReplace_34:output:0*#
_output_shapes
:?????????*
pattern did *
rewrite ?
StaticRegexReplace_36StaticRegexReplaceStaticRegexReplace_35:output:0*#
_output_shapes
:?????????*
pattern what *
rewrite ?
StaticRegexReplace_37StaticRegexReplaceStaticRegexReplace_36:output:0*#
_output_shapes
:?????????*
pattern in *
rewrite ?
StaticRegexReplace_38StaticRegexReplaceStaticRegexReplace_37:output:0*#
_output_shapes
:?????????*
pattern out *
rewrite ?
StaticRegexReplace_39StaticRegexReplaceStaticRegexReplace_38:output:0*#
_output_shapes
:?????????*
pattern than *
rewrite ?
StaticRegexReplace_40StaticRegexReplaceStaticRegexReplace_39:output:0*#
_output_shapes
:?????????*
pattern to *
rewrite ?
StaticRegexReplace_41StaticRegexReplaceStaticRegexReplace_40:output:0*#
_output_shapes
:?????????*
pattern	 because *
rewrite ?
StaticRegexReplace_42StaticRegexReplaceStaticRegexReplace_41:output:0*#
_output_shapes
:?????????*
pattern too *
rewrite ?
StaticRegexReplace_43StaticRegexReplaceStaticRegexReplace_42:output:0*#
_output_shapes
:?????????*
pattern here *
rewrite ?
StaticRegexReplace_44StaticRegexReplaceStaticRegexReplace_43:output:0*#
_output_shapes
:?????????*
pattern ma *
rewrite ?
StaticRegexReplace_45StaticRegexReplaceStaticRegexReplace_44:output:0*#
_output_shapes
:?????????*
pattern but *
rewrite ?
StaticRegexReplace_46StaticRegexReplaceStaticRegexReplace_45:output:0*#
_output_shapes
:?????????*
pattern
 before *
rewrite ?
StaticRegexReplace_47StaticRegexReplaceStaticRegexReplace_46:output:0*#
_output_shapes
:?????????*
pattern then *
rewrite ?
StaticRegexReplace_48StaticRegexReplaceStaticRegexReplace_47:output:0*#
_output_shapes
:?????????*
pattern
 should *
rewrite ?
StaticRegexReplace_49StaticRegexReplaceStaticRegexReplace_48:output:0*#
_output_shapes
:?????????*
pattern are *
rewrite ?
StaticRegexReplace_50StaticRegexReplaceStaticRegexReplace_49:output:0*#
_output_shapes
:?????????*
pattern had *
rewrite ?
StaticRegexReplace_51StaticRegexReplaceStaticRegexReplace_50:output:0*#
_output_shapes
:?????????*
pattern	 himself *
rewrite ?
StaticRegexReplace_52StaticRegexReplaceStaticRegexReplace_51:output:0*#
_output_shapes
:?????????*
pattern you *
rewrite ?
StaticRegexReplace_53StaticRegexReplaceStaticRegexReplace_52:output:0*#
_output_shapes
:?????????*
pattern
 yourself *
rewrite ?
StaticRegexReplace_54StaticRegexReplaceStaticRegexReplace_53:output:0*#
_output_shapes
:?????????*
pattern	 through *
rewrite ?
StaticRegexReplace_55StaticRegexReplaceStaticRegexReplace_54:output:0*#
_output_shapes
:?????????*
pattern hadn *
rewrite ?
StaticRegexReplace_56StaticRegexReplaceStaticRegexReplace_55:output:0*#
_output_shapes
:?????????*
pattern does *
rewrite ?
StaticRegexReplace_57StaticRegexReplaceStaticRegexReplace_56:output:0*#
_output_shapes
:?????????*
pattern m *
rewrite ?
StaticRegexReplace_58StaticRegexReplaceStaticRegexReplace_57:output:0*#
_output_shapes
:?????????*
pattern ain *
rewrite ?
StaticRegexReplace_59StaticRegexReplaceStaticRegexReplace_58:output:0*#
_output_shapes
:?????????*
pattern very *
rewrite ?
StaticRegexReplace_60StaticRegexReplaceStaticRegexReplace_59:output:0*#
_output_shapes
:?????????*
pattern	 weren't *
rewrite ?
StaticRegexReplace_61StaticRegexReplaceStaticRegexReplace_60:output:0*#
_output_shapes
:?????????*
pattern been *
rewrite ?
StaticRegexReplace_62StaticRegexReplaceStaticRegexReplace_61:output:0*#
_output_shapes
:?????????*
pattern will *
rewrite ?
StaticRegexReplace_63StaticRegexReplaceStaticRegexReplace_62:output:0*#
_output_shapes
:?????????*
pattern now *
rewrite ?
StaticRegexReplace_64StaticRegexReplaceStaticRegexReplace_63:output:0*#
_output_shapes
:?????????*
pattern they *
rewrite ?
StaticRegexReplace_65StaticRegexReplaceStaticRegexReplace_64:output:0*#
_output_shapes
:?????????*
pattern when *
rewrite ?
StaticRegexReplace_66StaticRegexReplaceStaticRegexReplace_65:output:0*#
_output_shapes
:?????????*
pattern was *
rewrite ?
StaticRegexReplace_67StaticRegexReplaceStaticRegexReplace_66:output:0*#
_output_shapes
:?????????*
pattern shouldn't *
rewrite ?
StaticRegexReplace_68StaticRegexReplaceStaticRegexReplace_67:output:0*#
_output_shapes
:?????????*
pattern	 herself *
rewrite ?
StaticRegexReplace_69StaticRegexReplaceStaticRegexReplace_68:output:0*#
_output_shapes
:?????????*
pattern	 above *
rewrite ?
StaticRegexReplace_70StaticRegexReplaceStaticRegexReplace_69:output:0*#
_output_shapes
:?????????*
pattern why *
rewrite ?
StaticRegexReplace_71StaticRegexReplaceStaticRegexReplace_70:output:0*#
_output_shapes
:?????????*
pattern her *
rewrite ?
StaticRegexReplace_72StaticRegexReplaceStaticRegexReplace_71:output:0*#
_output_shapes
:?????????*
pattern same *
rewrite ?
StaticRegexReplace_73StaticRegexReplaceStaticRegexReplace_72:output:0*#
_output_shapes
:?????????*
pattern
 having *
rewrite ?
StaticRegexReplace_74StaticRegexReplaceStaticRegexReplace_73:output:0*#
_output_shapes
:?????????*
pattern	 yours *
rewrite ?
StaticRegexReplace_75StaticRegexReplaceStaticRegexReplace_74:output:0*#
_output_shapes
:?????????*
pattern can *
rewrite ?
StaticRegexReplace_76StaticRegexReplaceStaticRegexReplace_75:output:0*#
_output_shapes
:?????????*
pattern
 wouldn't *
rewrite ?
StaticRegexReplace_77StaticRegexReplaceStaticRegexReplace_76:output:0*#
_output_shapes
:?????????*
pattern	 again *
rewrite ?
StaticRegexReplace_78StaticRegexReplaceStaticRegexReplace_77:output:0*#
_output_shapes
:?????????*
pattern do *
rewrite ?
StaticRegexReplace_79StaticRegexReplaceStaticRegexReplace_78:output:0*#
_output_shapes
:?????????*
pattern shan *
rewrite ?
StaticRegexReplace_80StaticRegexReplaceStaticRegexReplace_79:output:0*#
_output_shapes
:?????????*
pattern	 she's *
rewrite ?
StaticRegexReplace_81StaticRegexReplaceStaticRegexReplace_80:output:0*#
_output_shapes
:?????????*
pattern of *
rewrite ?
StaticRegexReplace_82StaticRegexReplaceStaticRegexReplace_81:output:0*#
_output_shapes
:?????????*
pattern	 against *
rewrite ?
StaticRegexReplace_83StaticRegexReplaceStaticRegexReplace_82:output:0*#
_output_shapes
:?????????*
pattern most *
rewrite ?
StaticRegexReplace_84StaticRegexReplaceStaticRegexReplace_83:output:0*#
_output_shapes
:?????????*
pattern	 isn't *
rewrite ?
StaticRegexReplace_85StaticRegexReplaceStaticRegexReplace_84:output:0*#
_output_shapes
:?????????*
pattern	 until *
rewrite ?
StaticRegexReplace_86StaticRegexReplaceStaticRegexReplace_85:output:0*#
_output_shapes
:?????????*
pattern it *
rewrite ?
StaticRegexReplace_87StaticRegexReplaceStaticRegexReplace_86:output:0*#
_output_shapes
:?????????*
pattern	 below *
rewrite ?
StaticRegexReplace_88StaticRegexReplaceStaticRegexReplace_87:output:0*#
_output_shapes
:?????????*
pattern	 mustn't *
rewrite ?
StaticRegexReplace_89StaticRegexReplaceStaticRegexReplace_88:output:0*#
_output_shapes
:?????????*
pattern by *
rewrite ?
StaticRegexReplace_90StaticRegexReplaceStaticRegexReplace_89:output:0*#
_output_shapes
:?????????*
pattern didn *
rewrite ?
StaticRegexReplace_91StaticRegexReplaceStaticRegexReplace_90:output:0*#
_output_shapes
:?????????*
pattern
 shan't *
rewrite ?
StaticRegexReplace_92StaticRegexReplaceStaticRegexReplace_91:output:0*#
_output_shapes
:?????????*
pattern who *
rewrite ?
StaticRegexReplace_93StaticRegexReplaceStaticRegexReplace_92:output:0*#
_output_shapes
:?????????*
pattern both *
rewrite ?
StaticRegexReplace_94StaticRegexReplaceStaticRegexReplace_93:output:0*#
_output_shapes
:?????????*
pattern re *
rewrite ?
StaticRegexReplace_95StaticRegexReplaceStaticRegexReplace_94:output:0*#
_output_shapes
:?????????*
pattern
 wouldn *
rewrite ?
StaticRegexReplace_96StaticRegexReplaceStaticRegexReplace_95:output:0*#
_output_shapes
:?????????*
pattern his *
rewrite ?
StaticRegexReplace_97StaticRegexReplaceStaticRegexReplace_96:output:0*#
_output_shapes
:?????????*
pattern ours *
rewrite ?
StaticRegexReplace_98StaticRegexReplaceStaticRegexReplace_97:output:0*#
_output_shapes
:?????????*
pattern
 itself *
rewrite ?
StaticRegexReplace_99StaticRegexReplaceStaticRegexReplace_98:output:0*#
_output_shapes
:?????????*
pattern don *
rewrite ?
StaticRegexReplace_100StaticRegexReplaceStaticRegexReplace_99:output:0*#
_output_shapes
:?????????*
pattern	 about *
rewrite ?
StaticRegexReplace_101StaticRegexReplaceStaticRegexReplace_100:output:0*#
_output_shapes
:?????????*
pattern o *
rewrite ?
StaticRegexReplace_102StaticRegexReplaceStaticRegexReplace_101:output:0*#
_output_shapes
:?????????*
pattern
 during *
rewrite ?
StaticRegexReplace_103StaticRegexReplaceStaticRegexReplace_102:output:0*#
_output_shapes
:?????????*
pattern whom *
rewrite ?
StaticRegexReplace_104StaticRegexReplaceStaticRegexReplace_103:output:0*#
_output_shapes
:?????????*
pattern
 mightn't *
rewrite ?
StaticRegexReplace_105StaticRegexReplaceStaticRegexReplace_104:output:0*#
_output_shapes
:?????????*
pattern
 didn't *
rewrite ?
StaticRegexReplace_106StaticRegexReplaceStaticRegexReplace_105:output:0*#
_output_shapes
:?????????*
pattern themselves *
rewrite ?
StaticRegexReplace_107StaticRegexReplaceStaticRegexReplace_106:output:0*#
_output_shapes
:?????????*
pattern with *
rewrite ?
StaticRegexReplace_108StaticRegexReplaceStaticRegexReplace_107:output:0*#
_output_shapes
:?????????*
pattern
 theirs *
rewrite ?
StaticRegexReplace_109StaticRegexReplaceStaticRegexReplace_108:output:0*#
_output_shapes
:?????????*
pattern	 further *
rewrite ?
StaticRegexReplace_110StaticRegexReplaceStaticRegexReplace_109:output:0*#
_output_shapes
:?????????*
pattern be *
rewrite ?
StaticRegexReplace_111StaticRegexReplaceStaticRegexReplace_110:output:0*#
_output_shapes
:?????????*
pattern	 weren *
rewrite ?
StaticRegexReplace_112StaticRegexReplaceStaticRegexReplace_111:output:0*#
_output_shapes
:?????????*
pattern own *
rewrite ?
StaticRegexReplace_113StaticRegexReplaceStaticRegexReplace_112:output:0*#
_output_shapes
:?????????*
pattern into *
rewrite ?
StaticRegexReplace_114StaticRegexReplaceStaticRegexReplace_113:output:0*#
_output_shapes
:?????????*
pattern t *
rewrite ?
StaticRegexReplace_115StaticRegexReplaceStaticRegexReplace_114:output:0*#
_output_shapes
:?????????*
pattern	 haven *
rewrite ?
StaticRegexReplace_116StaticRegexReplaceStaticRegexReplace_115:output:0*#
_output_shapes
:?????????*
pattern	 there *
rewrite ?
StaticRegexReplace_117StaticRegexReplaceStaticRegexReplace_116:output:0*#
_output_shapes
:?????????*
pattern yourselves *
rewrite ?
StaticRegexReplace_118StaticRegexReplaceStaticRegexReplace_117:output:0*#
_output_shapes
:?????????*
pattern
 aren't *
rewrite ?
StaticRegexReplace_119StaticRegexReplaceStaticRegexReplace_118:output:0*#
_output_shapes
:?????????*
pattern
 you'll *
rewrite ?
StaticRegexReplace_120StaticRegexReplaceStaticRegexReplace_119:output:0*#
_output_shapes
:?????????*
pattern how *
rewrite ?
StaticRegexReplace_121StaticRegexReplaceStaticRegexReplace_120:output:0*#
_output_shapes
:?????????*
pattern ourselves *
rewrite ?
StaticRegexReplace_122StaticRegexReplaceStaticRegexReplace_121:output:0*#
_output_shapes
:?????????*
pattern an *
rewrite ?
StaticRegexReplace_123StaticRegexReplaceStaticRegexReplace_122:output:0*#
_output_shapes
:?????????*
pattern	 don't *
rewrite ?
StaticRegexReplace_124StaticRegexReplaceStaticRegexReplace_123:output:0*#
_output_shapes
:?????????*
pattern	 doing *
rewrite ?
StaticRegexReplace_125StaticRegexReplaceStaticRegexReplace_124:output:0*#
_output_shapes
:?????????*
pattern more *
rewrite ?
StaticRegexReplace_126StaticRegexReplaceStaticRegexReplace_125:output:0*#
_output_shapes
:?????????*
pattern each *
rewrite ?
StaticRegexReplace_127StaticRegexReplaceStaticRegexReplace_126:output:0*#
_output_shapes
:?????????*
pattern we *
rewrite ?
StaticRegexReplace_128StaticRegexReplaceStaticRegexReplace_127:output:0*#
_output_shapes
:?????????*
pattern	 these *
rewrite ?
StaticRegexReplace_129StaticRegexReplaceStaticRegexReplace_128:output:0*#
_output_shapes
:?????????*
pattern over *
rewrite ?
StaticRegexReplace_130StaticRegexReplaceStaticRegexReplace_129:output:0*#
_output_shapes
:?????????*
pattern i *
rewrite ?
StaticRegexReplace_131StaticRegexReplaceStaticRegexReplace_130:output:0*#
_output_shapes
:?????????*
pattern nor *
rewrite ?
StaticRegexReplace_132StaticRegexReplaceStaticRegexReplace_131:output:0*#
_output_shapes
:?????????*
pattern	 needn't *
rewrite ?
StaticRegexReplace_133StaticRegexReplaceStaticRegexReplace_132:output:0*#
_output_shapes
:?????????*
pattern ll *
rewrite ?
StaticRegexReplace_134StaticRegexReplaceStaticRegexReplace_133:output:0*#
_output_shapes
:?????????*
pattern	 between *
rewrite ?
StaticRegexReplace_135StaticRegexReplaceStaticRegexReplace_134:output:0*#
_output_shapes
:?????????*
pattern should've *
rewrite ?
StaticRegexReplace_136StaticRegexReplaceStaticRegexReplace_135:output:0*#
_output_shapes
:?????????*
pattern
 hadn't *
rewrite ?
StaticRegexReplace_137StaticRegexReplaceStaticRegexReplace_136:output:0*#
_output_shapes
:?????????*
pattern hasn *
rewrite ?
StaticRegexReplace_138StaticRegexReplaceStaticRegexReplace_137:output:0*#
_output_shapes
:?????????*
pattern were *
rewrite ?
StaticRegexReplace_139StaticRegexReplaceStaticRegexReplace_138:output:0*#
_output_shapes
:?????????*
pattern has *
rewrite ?
StaticRegexReplace_140StaticRegexReplaceStaticRegexReplace_139:output:0*#
_output_shapes
:?????????*
pattern only *
rewrite ?
StaticRegexReplace_141StaticRegexReplaceStaticRegexReplace_140:output:0*#
_output_shapes
:?????????*
pattern she *
rewrite ?
StaticRegexReplace_142StaticRegexReplaceStaticRegexReplace_141:output:0*#
_output_shapes
:?????????*
pattern	 needn *
rewrite ?
StaticRegexReplace_143StaticRegexReplaceStaticRegexReplace_142:output:0*#
_output_shapes
:?????????*
pattern	 other *
rewrite ?
StaticRegexReplace_144StaticRegexReplaceStaticRegexReplace_143:output:0*#
_output_shapes
:?????????*
pattern
 hasn't *
rewrite ?
StaticRegexReplace_145StaticRegexReplaceStaticRegexReplace_144:output:0*#
_output_shapes
:?????????*
pattern a *
rewrite ?
StaticRegexReplace_146StaticRegexReplaceStaticRegexReplace_145:output:0*#
_output_shapes
:?????????*
pattern	 shouldn *
rewrite ?
StaticRegexReplace_147StaticRegexReplaceStaticRegexReplace_146:output:0*#
_output_shapes
:?????????*
pattern and *
rewrite ?
StaticRegexReplace_148StaticRegexReplaceStaticRegexReplace_147:output:0*#
_output_shapes
:?????????*
pattern	 those *
rewrite ?
StaticRegexReplace_149StaticRegexReplaceStaticRegexReplace_148:output:0*#
_output_shapes
:?????????*
pattern	 being *
rewrite ?
StaticRegexReplace_150StaticRegexReplaceStaticRegexReplace_149:output:0*#
_output_shapes
:?????????*
pattern such *
rewrite ?
StaticRegexReplace_151StaticRegexReplaceStaticRegexReplace_150:output:0*#
_output_shapes
:?????????*
pattern as *
rewrite ?
StaticRegexReplace_152StaticRegexReplaceStaticRegexReplace_151:output:0*#
_output_shapes
:?????????*
pattern ve *
rewrite ?
StaticRegexReplace_153StaticRegexReplaceStaticRegexReplace_152:output:0*#
_output_shapes
:?????????*
pattern hers *
rewrite ?
StaticRegexReplace_154StaticRegexReplaceStaticRegexReplace_153:output:0*#
_output_shapes
:?????????*
pattern s *
rewrite ?
StaticRegexReplace_155StaticRegexReplaceStaticRegexReplace_154:output:0*#
_output_shapes
:?????????*
pattern	 their *
rewrite ?
StaticRegexReplace_156StaticRegexReplaceStaticRegexReplace_155:output:0*#
_output_shapes
:?????????*
pattern	 haven't *
rewrite ?
StaticRegexReplace_157StaticRegexReplaceStaticRegexReplace_156:output:0*#
_output_shapes
:?????????*
pattern for *
rewrite ?
StaticRegexReplace_158StaticRegexReplaceStaticRegexReplace_157:output:0*#
_output_shapes
:?????????*
pattern if *
rewrite ?
StaticRegexReplace_159StaticRegexReplaceStaticRegexReplace_158:output:0*#
_output_shapes
:?????????*
pattern that *
rewrite ?
StaticRegexReplace_160StaticRegexReplaceStaticRegexReplace_159:output:0*#
_output_shapes
:?????????*
pattern isn *
rewrite ?
StaticRegexReplace_161StaticRegexReplaceStaticRegexReplace_160:output:0*#
_output_shapes
:?????????*
pattern him *
rewrite ?
StaticRegexReplace_162StaticRegexReplaceStaticRegexReplace_161:output:0*#
_output_shapes
:?????????*
pattern wasn *
rewrite ?
StaticRegexReplace_163StaticRegexReplaceStaticRegexReplace_162:output:0*#
_output_shapes
:?????????*
pattern any *
rewrite ?
StaticRegexReplace_164StaticRegexReplaceStaticRegexReplace_163:output:0*#
_output_shapes
:?????????*
pattern have *
rewrite ?
StaticRegexReplace_165StaticRegexReplaceStaticRegexReplace_164:output:0*#
_output_shapes
:?????????*
pattern	 under *
rewrite ?
StaticRegexReplace_166StaticRegexReplaceStaticRegexReplace_165:output:0*#
_output_shapes
:?????????*
pattern	 that'll *
rewrite ?
StaticRegexReplace_167StaticRegexReplaceStaticRegexReplace_166:output:0*#
_output_shapes
:?????????*
pattern or *
rewrite ?
StaticRegexReplace_168StaticRegexReplaceStaticRegexReplace_167:output:0*#
_output_shapes
:?????????*
pattern no *
rewrite ?
StaticRegexReplace_169StaticRegexReplaceStaticRegexReplace_168:output:0*#
_output_shapes
:?????????*
pattern he *
rewrite ?
StaticRegexReplace_170StaticRegexReplaceStaticRegexReplace_169:output:0*#
_output_shapes
:?????????*
pattern
 you're *
rewrite ?
StaticRegexReplace_171StaticRegexReplaceStaticRegexReplace_170:output:0*#
_output_shapes
:?????????*
pattern this *
rewrite ?
StaticRegexReplace_172StaticRegexReplaceStaticRegexReplace_171:output:0*#
_output_shapes
:?????????*
pattern	 doesn *
rewrite ?
StaticRegexReplace_173StaticRegexReplaceStaticRegexReplace_172:output:0*#
_output_shapes
:?????????*
pattern	 you'd *
rewrite ?
StaticRegexReplace_174StaticRegexReplaceStaticRegexReplace_173:output:0*#
_output_shapes
:?????????*
pattern up *
rewrite ?
StaticRegexReplace_175StaticRegexReplaceStaticRegexReplace_174:output:0*#
_output_shapes
:?????????*
pattern
 you've *
rewrite ?
StaticRegexReplace_176StaticRegexReplaceStaticRegexReplace_175:output:0*#
_output_shapes
:?????????*
pattern your *
rewrite ?
StaticRegexReplace_177StaticRegexReplaceStaticRegexReplace_176:output:0*#
_output_shapes
:?????????*
pattern at *
rewrite ?
StaticRegexReplace_178StaticRegexReplaceStaticRegexReplace_177:output:0*#
_output_shapes
:?????????*
pattern few *
rewrite ?
StaticRegexReplace_179StaticRegexReplaceStaticRegexReplace_178:output:0*#
_output_shapes
:?????????*
pattern its *
rewrite ?
StaticRegexReplace_180StaticRegexReplaceStaticRegexReplace_179:output:0*#
_output_shapes
:?????????*
pattern y *
rewrite ?
StaticRegexReplace_181StaticRegexReplaceStaticRegexReplace_180:output:0*#
_output_shapes
:?????????*
pattern down *
rewrite ?
StaticRegexReplace_182StaticRegexReplaceStaticRegexReplace_181:output:0*#
_output_shapes
:?????????*A
pattern64[!"\#\$%\&'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}\~]*
rewrite R
StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
StringSplit/StringSplitV2StringSplitV2StaticRegexReplace_182:output:0StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:p
StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
StringSplit/strided_sliceStridedSlice#StringSplit/StringSplitV2:indices:0(StringSplit/strided_slice/stack:output:0*StringSplit/strided_slice/stack_1:output:0*StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_maskk
!StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
StringSplit/strided_slice_1StridedSlice!StringSplit/StringSplitV2:shape:0*StringSplit/strided_slice_1/stack:output:0,StringSplit/strided_slice_1/stack_1:output:0,StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
BStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast"StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast$StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdUStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
PStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterTStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0YStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastRStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2SStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulOStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
OStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumVStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
MStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2VStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
UniqueWithCountsUniqueWithCounts"StringSplit/StringSplitV2:values:0*
T0*A
_output_shapes/
-:?????????:?????????:?????????*
out_idx0	?
(None_lookup_table_find/LookupTableFindV2LookupTableFindV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:06none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
:|
addAddV2UniqueWithCounts:count:01None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:?
,None_lookup_table_insert/LookupTableInsertV2LookupTableInsertV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:0add:z:0)^None_lookup_table_find/LookupTableFindV2",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 *(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2"
IteratorGetNextIteratorGetNext2T
(None_lookup_table_find/LookupTableFindV2(None_lookup_table_find/LookupTableFindV22\
,None_lookup_table_insert/LookupTableInsertV2,None_lookup_table_insert/LookupTableInsertV2:( $
"
_user_specified_name
iterator:

_output_shapes
: 
?

?
#__inference_signature_wrapper_78808
text_vectorization_input
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?N 
	unknown_4:  
	unknown_5: 
	unknown_6: 
	unknown_7:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalltext_vectorization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
	*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_77588o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
#
_output_shapes
:?????????
2
_user_specified_nametext_vectorization_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?

?
*__inference_sequential_layer_call_fn_78281
text_vectorization_input
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?N 
	unknown_4:  
	unknown_5: 
	unknown_6: 
	unknown_7:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalltext_vectorization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
	*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_78237o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
#
_output_shapes
:?????????
2
_user_specified_nametext_vectorization_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?

?
@__inference_dense_layer_call_and_return_conditional_losses_77862

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
.
__inference__initializer_79719
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?

?
*__inference_sequential_layer_call_fn_77913
text_vectorization_input
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?N 
	unknown_4:  
	unknown_5: 
	unknown_6: 
	unknown_7:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalltext_vectorization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
	*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_77892o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
#
_output_shapes
:?????????
2
_user_specified_nametext_vectorization_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
%__inference_dense_layer_call_fn_79634

inputs
unknown:  
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_77862o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
o
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_77598

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?g
?
!__inference__traced_restore_79958
file_prefix8
%assignvariableop_embedding_embeddings:	?N 1
assignvariableop_1_dense_kernel:  +
assignvariableop_2_dense_bias: 3
!assignvariableop_3_dense_1_kernel: -
assignvariableop_4_dense_1_bias:&
assignvariableop_5_adam_iter:	 (
assignvariableop_6_adam_beta_1: (
assignvariableop_7_adam_beta_2: '
assignvariableop_8_adam_decay: /
%assignvariableop_9_adam_learning_rate: M
Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtable: %
assignvariableop_10_total_1: %
assignvariableop_11_count_1: #
assignvariableop_12_total: #
assignvariableop_13_count: B
/assignvariableop_14_adam_embedding_embeddings_m:	?N 9
'assignvariableop_15_adam_dense_kernel_m:  3
%assignvariableop_16_adam_dense_bias_m: ;
)assignvariableop_17_adam_dense_1_kernel_m: 5
'assignvariableop_18_adam_dense_1_bias_m:B
/assignvariableop_19_adam_embedding_embeddings_v:	?N 9
'assignvariableop_20_adam_dense_kernel_v:  3
%assignvariableop_21_adam_dense_bias_v: ;
)assignvariableop_22_adam_dense_1_kernel_v: 5
'assignvariableop_23_adam_dense_1_bias_v:
identity_25??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?2MutableHashTable_table_restore/LookupTableImportV2?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-valuesB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapesn
l:::::::::::::::::::::::::::*)
dtypes
2		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp%assignvariableop_embedding_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_1_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_1_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_iterIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_1Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_2Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_decayIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp%assignvariableop_9_adam_learning_rateIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0?
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtableRestoreV2:tensors:10RestoreV2:tensors:11*	
Tin0*

Tout0	*#
_class
loc:@MutableHashTable*
_output_shapes
 _
Identity_10IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpassignvariableop_10_total_1Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpassignvariableop_11_count_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpassignvariableop_12_totalIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_countIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp/assignvariableop_14_adam_embedding_embeddings_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp'assignvariableop_15_adam_dense_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp%assignvariableop_16_adam_dense_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_dense_1_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp'assignvariableop_18_adam_dense_1_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp/assignvariableop_19_adam_embedding_embeddings_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp'assignvariableop_20_adam_dense_kernel_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp%assignvariableop_21_adam_dense_bias_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_dense_1_kernel_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp'assignvariableop_23_adam_dense_1_bias_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_93^MutableHashTable_table_restore/LookupTableImportV2^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_25IdentityIdentity_24:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_93^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "#
identity_25Identity_25:output:0*G
_input_shapes6
4: : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:)%
#
_class
loc:@MutableHashTable
??
?
E__inference_sequential_layer_call_and_return_conditional_losses_78237

inputsO
Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	"
embedding_78221:	?N 
dense_78225:  
dense_78227: 
dense_1_78231: 
dense_1_78233:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!embedding/StatefulPartitionedCall?>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2Z
text_vectorization/StringLowerStringLowerinputs*#
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*#
_output_shapes
:?????????*
pattern<br />*
rewrite ?
'text_vectorization/StaticRegexReplace_1StaticRegexReplace.text_vectorization/StaticRegexReplace:output:0*#
_output_shapes
:?????????*+
pattern \d+(?:\.\d*)?(?:[eE][+-]?\d+)?*
rewrite ?
'text_vectorization/StaticRegexReplace_2StaticRegexReplace0text_vectorization/StaticRegexReplace_1:output:0*#
_output_shapes
:?????????*
pattern@([A-Za-z0-9_]+)*
rewrite ?
'text_vectorization/StaticRegexReplace_3StaticRegexReplace0text_vectorization/StaticRegexReplace_2:output:0*#
_output_shapes
:?????????*
pattern	 which *
rewrite ?
'text_vectorization/StaticRegexReplace_4StaticRegexReplace0text_vectorization/StaticRegexReplace_3:output:0*#
_output_shapes
:?????????*
pattern
 couldn *
rewrite ?
'text_vectorization/StaticRegexReplace_5StaticRegexReplace0text_vectorization/StaticRegexReplace_4:output:0*#
_output_shapes
:?????????*
pattern once *
rewrite ?
'text_vectorization/StaticRegexReplace_6StaticRegexReplace0text_vectorization/StaticRegexReplace_5:output:0*#
_output_shapes
:?????????*
pattern is *
rewrite ?
'text_vectorization/StaticRegexReplace_7StaticRegexReplace0text_vectorization/StaticRegexReplace_6:output:0*#
_output_shapes
:?????????*
pattern on *
rewrite ?
'text_vectorization/StaticRegexReplace_8StaticRegexReplace0text_vectorization/StaticRegexReplace_7:output:0*#
_output_shapes
:?????????*
pattern some *
rewrite ?
'text_vectorization/StaticRegexReplace_9StaticRegexReplace0text_vectorization/StaticRegexReplace_8:output:0*#
_output_shapes
:?????????*
pattern not *
rewrite ?
(text_vectorization/StaticRegexReplace_10StaticRegexReplace0text_vectorization/StaticRegexReplace_9:output:0*#
_output_shapes
:?????????*
pattern won *
rewrite ?
(text_vectorization/StaticRegexReplace_11StaticRegexReplace1text_vectorization/StaticRegexReplace_10:output:0*#
_output_shapes
:?????????*
pattern	 while *
rewrite ?
(text_vectorization/StaticRegexReplace_12StaticRegexReplace1text_vectorization/StaticRegexReplace_11:output:0*#
_output_shapes
:?????????*
pattern them *
rewrite ?
(text_vectorization/StaticRegexReplace_13StaticRegexReplace1text_vectorization/StaticRegexReplace_12:output:0*#
_output_shapes
:?????????*
pattern am *
rewrite ?
(text_vectorization/StaticRegexReplace_14StaticRegexReplace1text_vectorization/StaticRegexReplace_13:output:0*#
_output_shapes
:?????????*
pattern	 where *
rewrite ?
(text_vectorization/StaticRegexReplace_15StaticRegexReplace1text_vectorization/StaticRegexReplace_14:output:0*#
_output_shapes
:?????????*
pattern my *
rewrite ?
(text_vectorization/StaticRegexReplace_16StaticRegexReplace1text_vectorization/StaticRegexReplace_15:output:0*#
_output_shapes
:?????????*
pattern me *
rewrite ?
(text_vectorization/StaticRegexReplace_17StaticRegexReplace1text_vectorization/StaticRegexReplace_16:output:0*#
_output_shapes
:?????????*
pattern
 couldn't *
rewrite ?
(text_vectorization/StaticRegexReplace_18StaticRegexReplace1text_vectorization/StaticRegexReplace_17:output:0*#
_output_shapes
:?????????*
pattern all *
rewrite ?
(text_vectorization/StaticRegexReplace_19StaticRegexReplace1text_vectorization/StaticRegexReplace_18:output:0*#
_output_shapes
:?????????*
pattern it's *
rewrite ?
(text_vectorization/StaticRegexReplace_20StaticRegexReplace1text_vectorization/StaticRegexReplace_19:output:0*#
_output_shapes
:?????????*
pattern off *
rewrite ?
(text_vectorization/StaticRegexReplace_21StaticRegexReplace1text_vectorization/StaticRegexReplace_20:output:0*#
_output_shapes
:?????????*
pattern so *
rewrite ?
(text_vectorization/StaticRegexReplace_22StaticRegexReplace1text_vectorization/StaticRegexReplace_21:output:0*#
_output_shapes
:?????????*
pattern
 mightn *
rewrite ?
(text_vectorization/StaticRegexReplace_23StaticRegexReplace1text_vectorization/StaticRegexReplace_22:output:0*#
_output_shapes
:?????????*
pattern our *
rewrite ?
(text_vectorization/StaticRegexReplace_24StaticRegexReplace1text_vectorization/StaticRegexReplace_23:output:0*#
_output_shapes
:?????????*
pattern aren *
rewrite ?
(text_vectorization/StaticRegexReplace_25StaticRegexReplace1text_vectorization/StaticRegexReplace_24:output:0*#
_output_shapes
:?????????*
pattern	 won't *
rewrite ?
(text_vectorization/StaticRegexReplace_26StaticRegexReplace1text_vectorization/StaticRegexReplace_25:output:0*#
_output_shapes
:?????????*
pattern the *
rewrite ?
(text_vectorization/StaticRegexReplace_27StaticRegexReplace1text_vectorization/StaticRegexReplace_26:output:0*#
_output_shapes
:?????????*
pattern
 wasn't *
rewrite ?
(text_vectorization/StaticRegexReplace_28StaticRegexReplace1text_vectorization/StaticRegexReplace_27:output:0*#
_output_shapes
:?????????*
pattern just *
rewrite ?
(text_vectorization/StaticRegexReplace_29StaticRegexReplace1text_vectorization/StaticRegexReplace_28:output:0*#
_output_shapes
:?????????*
pattern
 myself *
rewrite ?
(text_vectorization/StaticRegexReplace_30StaticRegexReplace1text_vectorization/StaticRegexReplace_29:output:0*#
_output_shapes
:?????????*
pattern	 after *
rewrite ?
(text_vectorization/StaticRegexReplace_31StaticRegexReplace1text_vectorization/StaticRegexReplace_30:output:0*#
_output_shapes
:?????????*
pattern from *
rewrite ?
(text_vectorization/StaticRegexReplace_32StaticRegexReplace1text_vectorization/StaticRegexReplace_31:output:0*#
_output_shapes
:?????????*
pattern d *
rewrite ?
(text_vectorization/StaticRegexReplace_33StaticRegexReplace1text_vectorization/StaticRegexReplace_32:output:0*#
_output_shapes
:?????????*
pattern	 mustn *
rewrite ?
(text_vectorization/StaticRegexReplace_34StaticRegexReplace1text_vectorization/StaticRegexReplace_33:output:0*#
_output_shapes
:?????????*
pattern	 doesn't *
rewrite ?
(text_vectorization/StaticRegexReplace_35StaticRegexReplace1text_vectorization/StaticRegexReplace_34:output:0*#
_output_shapes
:?????????*
pattern did *
rewrite ?
(text_vectorization/StaticRegexReplace_36StaticRegexReplace1text_vectorization/StaticRegexReplace_35:output:0*#
_output_shapes
:?????????*
pattern what *
rewrite ?
(text_vectorization/StaticRegexReplace_37StaticRegexReplace1text_vectorization/StaticRegexReplace_36:output:0*#
_output_shapes
:?????????*
pattern in *
rewrite ?
(text_vectorization/StaticRegexReplace_38StaticRegexReplace1text_vectorization/StaticRegexReplace_37:output:0*#
_output_shapes
:?????????*
pattern out *
rewrite ?
(text_vectorization/StaticRegexReplace_39StaticRegexReplace1text_vectorization/StaticRegexReplace_38:output:0*#
_output_shapes
:?????????*
pattern than *
rewrite ?
(text_vectorization/StaticRegexReplace_40StaticRegexReplace1text_vectorization/StaticRegexReplace_39:output:0*#
_output_shapes
:?????????*
pattern to *
rewrite ?
(text_vectorization/StaticRegexReplace_41StaticRegexReplace1text_vectorization/StaticRegexReplace_40:output:0*#
_output_shapes
:?????????*
pattern	 because *
rewrite ?
(text_vectorization/StaticRegexReplace_42StaticRegexReplace1text_vectorization/StaticRegexReplace_41:output:0*#
_output_shapes
:?????????*
pattern too *
rewrite ?
(text_vectorization/StaticRegexReplace_43StaticRegexReplace1text_vectorization/StaticRegexReplace_42:output:0*#
_output_shapes
:?????????*
pattern here *
rewrite ?
(text_vectorization/StaticRegexReplace_44StaticRegexReplace1text_vectorization/StaticRegexReplace_43:output:0*#
_output_shapes
:?????????*
pattern ma *
rewrite ?
(text_vectorization/StaticRegexReplace_45StaticRegexReplace1text_vectorization/StaticRegexReplace_44:output:0*#
_output_shapes
:?????????*
pattern but *
rewrite ?
(text_vectorization/StaticRegexReplace_46StaticRegexReplace1text_vectorization/StaticRegexReplace_45:output:0*#
_output_shapes
:?????????*
pattern
 before *
rewrite ?
(text_vectorization/StaticRegexReplace_47StaticRegexReplace1text_vectorization/StaticRegexReplace_46:output:0*#
_output_shapes
:?????????*
pattern then *
rewrite ?
(text_vectorization/StaticRegexReplace_48StaticRegexReplace1text_vectorization/StaticRegexReplace_47:output:0*#
_output_shapes
:?????????*
pattern
 should *
rewrite ?
(text_vectorization/StaticRegexReplace_49StaticRegexReplace1text_vectorization/StaticRegexReplace_48:output:0*#
_output_shapes
:?????????*
pattern are *
rewrite ?
(text_vectorization/StaticRegexReplace_50StaticRegexReplace1text_vectorization/StaticRegexReplace_49:output:0*#
_output_shapes
:?????????*
pattern had *
rewrite ?
(text_vectorization/StaticRegexReplace_51StaticRegexReplace1text_vectorization/StaticRegexReplace_50:output:0*#
_output_shapes
:?????????*
pattern	 himself *
rewrite ?
(text_vectorization/StaticRegexReplace_52StaticRegexReplace1text_vectorization/StaticRegexReplace_51:output:0*#
_output_shapes
:?????????*
pattern you *
rewrite ?
(text_vectorization/StaticRegexReplace_53StaticRegexReplace1text_vectorization/StaticRegexReplace_52:output:0*#
_output_shapes
:?????????*
pattern
 yourself *
rewrite ?
(text_vectorization/StaticRegexReplace_54StaticRegexReplace1text_vectorization/StaticRegexReplace_53:output:0*#
_output_shapes
:?????????*
pattern	 through *
rewrite ?
(text_vectorization/StaticRegexReplace_55StaticRegexReplace1text_vectorization/StaticRegexReplace_54:output:0*#
_output_shapes
:?????????*
pattern hadn *
rewrite ?
(text_vectorization/StaticRegexReplace_56StaticRegexReplace1text_vectorization/StaticRegexReplace_55:output:0*#
_output_shapes
:?????????*
pattern does *
rewrite ?
(text_vectorization/StaticRegexReplace_57StaticRegexReplace1text_vectorization/StaticRegexReplace_56:output:0*#
_output_shapes
:?????????*
pattern m *
rewrite ?
(text_vectorization/StaticRegexReplace_58StaticRegexReplace1text_vectorization/StaticRegexReplace_57:output:0*#
_output_shapes
:?????????*
pattern ain *
rewrite ?
(text_vectorization/StaticRegexReplace_59StaticRegexReplace1text_vectorization/StaticRegexReplace_58:output:0*#
_output_shapes
:?????????*
pattern very *
rewrite ?
(text_vectorization/StaticRegexReplace_60StaticRegexReplace1text_vectorization/StaticRegexReplace_59:output:0*#
_output_shapes
:?????????*
pattern	 weren't *
rewrite ?
(text_vectorization/StaticRegexReplace_61StaticRegexReplace1text_vectorization/StaticRegexReplace_60:output:0*#
_output_shapes
:?????????*
pattern been *
rewrite ?
(text_vectorization/StaticRegexReplace_62StaticRegexReplace1text_vectorization/StaticRegexReplace_61:output:0*#
_output_shapes
:?????????*
pattern will *
rewrite ?
(text_vectorization/StaticRegexReplace_63StaticRegexReplace1text_vectorization/StaticRegexReplace_62:output:0*#
_output_shapes
:?????????*
pattern now *
rewrite ?
(text_vectorization/StaticRegexReplace_64StaticRegexReplace1text_vectorization/StaticRegexReplace_63:output:0*#
_output_shapes
:?????????*
pattern they *
rewrite ?
(text_vectorization/StaticRegexReplace_65StaticRegexReplace1text_vectorization/StaticRegexReplace_64:output:0*#
_output_shapes
:?????????*
pattern when *
rewrite ?
(text_vectorization/StaticRegexReplace_66StaticRegexReplace1text_vectorization/StaticRegexReplace_65:output:0*#
_output_shapes
:?????????*
pattern was *
rewrite ?
(text_vectorization/StaticRegexReplace_67StaticRegexReplace1text_vectorization/StaticRegexReplace_66:output:0*#
_output_shapes
:?????????*
pattern shouldn't *
rewrite ?
(text_vectorization/StaticRegexReplace_68StaticRegexReplace1text_vectorization/StaticRegexReplace_67:output:0*#
_output_shapes
:?????????*
pattern	 herself *
rewrite ?
(text_vectorization/StaticRegexReplace_69StaticRegexReplace1text_vectorization/StaticRegexReplace_68:output:0*#
_output_shapes
:?????????*
pattern	 above *
rewrite ?
(text_vectorization/StaticRegexReplace_70StaticRegexReplace1text_vectorization/StaticRegexReplace_69:output:0*#
_output_shapes
:?????????*
pattern why *
rewrite ?
(text_vectorization/StaticRegexReplace_71StaticRegexReplace1text_vectorization/StaticRegexReplace_70:output:0*#
_output_shapes
:?????????*
pattern her *
rewrite ?
(text_vectorization/StaticRegexReplace_72StaticRegexReplace1text_vectorization/StaticRegexReplace_71:output:0*#
_output_shapes
:?????????*
pattern same *
rewrite ?
(text_vectorization/StaticRegexReplace_73StaticRegexReplace1text_vectorization/StaticRegexReplace_72:output:0*#
_output_shapes
:?????????*
pattern
 having *
rewrite ?
(text_vectorization/StaticRegexReplace_74StaticRegexReplace1text_vectorization/StaticRegexReplace_73:output:0*#
_output_shapes
:?????????*
pattern	 yours *
rewrite ?
(text_vectorization/StaticRegexReplace_75StaticRegexReplace1text_vectorization/StaticRegexReplace_74:output:0*#
_output_shapes
:?????????*
pattern can *
rewrite ?
(text_vectorization/StaticRegexReplace_76StaticRegexReplace1text_vectorization/StaticRegexReplace_75:output:0*#
_output_shapes
:?????????*
pattern
 wouldn't *
rewrite ?
(text_vectorization/StaticRegexReplace_77StaticRegexReplace1text_vectorization/StaticRegexReplace_76:output:0*#
_output_shapes
:?????????*
pattern	 again *
rewrite ?
(text_vectorization/StaticRegexReplace_78StaticRegexReplace1text_vectorization/StaticRegexReplace_77:output:0*#
_output_shapes
:?????????*
pattern do *
rewrite ?
(text_vectorization/StaticRegexReplace_79StaticRegexReplace1text_vectorization/StaticRegexReplace_78:output:0*#
_output_shapes
:?????????*
pattern shan *
rewrite ?
(text_vectorization/StaticRegexReplace_80StaticRegexReplace1text_vectorization/StaticRegexReplace_79:output:0*#
_output_shapes
:?????????*
pattern	 she's *
rewrite ?
(text_vectorization/StaticRegexReplace_81StaticRegexReplace1text_vectorization/StaticRegexReplace_80:output:0*#
_output_shapes
:?????????*
pattern of *
rewrite ?
(text_vectorization/StaticRegexReplace_82StaticRegexReplace1text_vectorization/StaticRegexReplace_81:output:0*#
_output_shapes
:?????????*
pattern	 against *
rewrite ?
(text_vectorization/StaticRegexReplace_83StaticRegexReplace1text_vectorization/StaticRegexReplace_82:output:0*#
_output_shapes
:?????????*
pattern most *
rewrite ?
(text_vectorization/StaticRegexReplace_84StaticRegexReplace1text_vectorization/StaticRegexReplace_83:output:0*#
_output_shapes
:?????????*
pattern	 isn't *
rewrite ?
(text_vectorization/StaticRegexReplace_85StaticRegexReplace1text_vectorization/StaticRegexReplace_84:output:0*#
_output_shapes
:?????????*
pattern	 until *
rewrite ?
(text_vectorization/StaticRegexReplace_86StaticRegexReplace1text_vectorization/StaticRegexReplace_85:output:0*#
_output_shapes
:?????????*
pattern it *
rewrite ?
(text_vectorization/StaticRegexReplace_87StaticRegexReplace1text_vectorization/StaticRegexReplace_86:output:0*#
_output_shapes
:?????????*
pattern	 below *
rewrite ?
(text_vectorization/StaticRegexReplace_88StaticRegexReplace1text_vectorization/StaticRegexReplace_87:output:0*#
_output_shapes
:?????????*
pattern	 mustn't *
rewrite ?
(text_vectorization/StaticRegexReplace_89StaticRegexReplace1text_vectorization/StaticRegexReplace_88:output:0*#
_output_shapes
:?????????*
pattern by *
rewrite ?
(text_vectorization/StaticRegexReplace_90StaticRegexReplace1text_vectorization/StaticRegexReplace_89:output:0*#
_output_shapes
:?????????*
pattern didn *
rewrite ?
(text_vectorization/StaticRegexReplace_91StaticRegexReplace1text_vectorization/StaticRegexReplace_90:output:0*#
_output_shapes
:?????????*
pattern
 shan't *
rewrite ?
(text_vectorization/StaticRegexReplace_92StaticRegexReplace1text_vectorization/StaticRegexReplace_91:output:0*#
_output_shapes
:?????????*
pattern who *
rewrite ?
(text_vectorization/StaticRegexReplace_93StaticRegexReplace1text_vectorization/StaticRegexReplace_92:output:0*#
_output_shapes
:?????????*
pattern both *
rewrite ?
(text_vectorization/StaticRegexReplace_94StaticRegexReplace1text_vectorization/StaticRegexReplace_93:output:0*#
_output_shapes
:?????????*
pattern re *
rewrite ?
(text_vectorization/StaticRegexReplace_95StaticRegexReplace1text_vectorization/StaticRegexReplace_94:output:0*#
_output_shapes
:?????????*
pattern
 wouldn *
rewrite ?
(text_vectorization/StaticRegexReplace_96StaticRegexReplace1text_vectorization/StaticRegexReplace_95:output:0*#
_output_shapes
:?????????*
pattern his *
rewrite ?
(text_vectorization/StaticRegexReplace_97StaticRegexReplace1text_vectorization/StaticRegexReplace_96:output:0*#
_output_shapes
:?????????*
pattern ours *
rewrite ?
(text_vectorization/StaticRegexReplace_98StaticRegexReplace1text_vectorization/StaticRegexReplace_97:output:0*#
_output_shapes
:?????????*
pattern
 itself *
rewrite ?
(text_vectorization/StaticRegexReplace_99StaticRegexReplace1text_vectorization/StaticRegexReplace_98:output:0*#
_output_shapes
:?????????*
pattern don *
rewrite ?
)text_vectorization/StaticRegexReplace_100StaticRegexReplace1text_vectorization/StaticRegexReplace_99:output:0*#
_output_shapes
:?????????*
pattern	 about *
rewrite ?
)text_vectorization/StaticRegexReplace_101StaticRegexReplace2text_vectorization/StaticRegexReplace_100:output:0*#
_output_shapes
:?????????*
pattern o *
rewrite ?
)text_vectorization/StaticRegexReplace_102StaticRegexReplace2text_vectorization/StaticRegexReplace_101:output:0*#
_output_shapes
:?????????*
pattern
 during *
rewrite ?
)text_vectorization/StaticRegexReplace_103StaticRegexReplace2text_vectorization/StaticRegexReplace_102:output:0*#
_output_shapes
:?????????*
pattern whom *
rewrite ?
)text_vectorization/StaticRegexReplace_104StaticRegexReplace2text_vectorization/StaticRegexReplace_103:output:0*#
_output_shapes
:?????????*
pattern
 mightn't *
rewrite ?
)text_vectorization/StaticRegexReplace_105StaticRegexReplace2text_vectorization/StaticRegexReplace_104:output:0*#
_output_shapes
:?????????*
pattern
 didn't *
rewrite ?
)text_vectorization/StaticRegexReplace_106StaticRegexReplace2text_vectorization/StaticRegexReplace_105:output:0*#
_output_shapes
:?????????*
pattern themselves *
rewrite ?
)text_vectorization/StaticRegexReplace_107StaticRegexReplace2text_vectorization/StaticRegexReplace_106:output:0*#
_output_shapes
:?????????*
pattern with *
rewrite ?
)text_vectorization/StaticRegexReplace_108StaticRegexReplace2text_vectorization/StaticRegexReplace_107:output:0*#
_output_shapes
:?????????*
pattern
 theirs *
rewrite ?
)text_vectorization/StaticRegexReplace_109StaticRegexReplace2text_vectorization/StaticRegexReplace_108:output:0*#
_output_shapes
:?????????*
pattern	 further *
rewrite ?
)text_vectorization/StaticRegexReplace_110StaticRegexReplace2text_vectorization/StaticRegexReplace_109:output:0*#
_output_shapes
:?????????*
pattern be *
rewrite ?
)text_vectorization/StaticRegexReplace_111StaticRegexReplace2text_vectorization/StaticRegexReplace_110:output:0*#
_output_shapes
:?????????*
pattern	 weren *
rewrite ?
)text_vectorization/StaticRegexReplace_112StaticRegexReplace2text_vectorization/StaticRegexReplace_111:output:0*#
_output_shapes
:?????????*
pattern own *
rewrite ?
)text_vectorization/StaticRegexReplace_113StaticRegexReplace2text_vectorization/StaticRegexReplace_112:output:0*#
_output_shapes
:?????????*
pattern into *
rewrite ?
)text_vectorization/StaticRegexReplace_114StaticRegexReplace2text_vectorization/StaticRegexReplace_113:output:0*#
_output_shapes
:?????????*
pattern t *
rewrite ?
)text_vectorization/StaticRegexReplace_115StaticRegexReplace2text_vectorization/StaticRegexReplace_114:output:0*#
_output_shapes
:?????????*
pattern	 haven *
rewrite ?
)text_vectorization/StaticRegexReplace_116StaticRegexReplace2text_vectorization/StaticRegexReplace_115:output:0*#
_output_shapes
:?????????*
pattern	 there *
rewrite ?
)text_vectorization/StaticRegexReplace_117StaticRegexReplace2text_vectorization/StaticRegexReplace_116:output:0*#
_output_shapes
:?????????*
pattern yourselves *
rewrite ?
)text_vectorization/StaticRegexReplace_118StaticRegexReplace2text_vectorization/StaticRegexReplace_117:output:0*#
_output_shapes
:?????????*
pattern
 aren't *
rewrite ?
)text_vectorization/StaticRegexReplace_119StaticRegexReplace2text_vectorization/StaticRegexReplace_118:output:0*#
_output_shapes
:?????????*
pattern
 you'll *
rewrite ?
)text_vectorization/StaticRegexReplace_120StaticRegexReplace2text_vectorization/StaticRegexReplace_119:output:0*#
_output_shapes
:?????????*
pattern how *
rewrite ?
)text_vectorization/StaticRegexReplace_121StaticRegexReplace2text_vectorization/StaticRegexReplace_120:output:0*#
_output_shapes
:?????????*
pattern ourselves *
rewrite ?
)text_vectorization/StaticRegexReplace_122StaticRegexReplace2text_vectorization/StaticRegexReplace_121:output:0*#
_output_shapes
:?????????*
pattern an *
rewrite ?
)text_vectorization/StaticRegexReplace_123StaticRegexReplace2text_vectorization/StaticRegexReplace_122:output:0*#
_output_shapes
:?????????*
pattern	 don't *
rewrite ?
)text_vectorization/StaticRegexReplace_124StaticRegexReplace2text_vectorization/StaticRegexReplace_123:output:0*#
_output_shapes
:?????????*
pattern	 doing *
rewrite ?
)text_vectorization/StaticRegexReplace_125StaticRegexReplace2text_vectorization/StaticRegexReplace_124:output:0*#
_output_shapes
:?????????*
pattern more *
rewrite ?
)text_vectorization/StaticRegexReplace_126StaticRegexReplace2text_vectorization/StaticRegexReplace_125:output:0*#
_output_shapes
:?????????*
pattern each *
rewrite ?
)text_vectorization/StaticRegexReplace_127StaticRegexReplace2text_vectorization/StaticRegexReplace_126:output:0*#
_output_shapes
:?????????*
pattern we *
rewrite ?
)text_vectorization/StaticRegexReplace_128StaticRegexReplace2text_vectorization/StaticRegexReplace_127:output:0*#
_output_shapes
:?????????*
pattern	 these *
rewrite ?
)text_vectorization/StaticRegexReplace_129StaticRegexReplace2text_vectorization/StaticRegexReplace_128:output:0*#
_output_shapes
:?????????*
pattern over *
rewrite ?
)text_vectorization/StaticRegexReplace_130StaticRegexReplace2text_vectorization/StaticRegexReplace_129:output:0*#
_output_shapes
:?????????*
pattern i *
rewrite ?
)text_vectorization/StaticRegexReplace_131StaticRegexReplace2text_vectorization/StaticRegexReplace_130:output:0*#
_output_shapes
:?????????*
pattern nor *
rewrite ?
)text_vectorization/StaticRegexReplace_132StaticRegexReplace2text_vectorization/StaticRegexReplace_131:output:0*#
_output_shapes
:?????????*
pattern	 needn't *
rewrite ?
)text_vectorization/StaticRegexReplace_133StaticRegexReplace2text_vectorization/StaticRegexReplace_132:output:0*#
_output_shapes
:?????????*
pattern ll *
rewrite ?
)text_vectorization/StaticRegexReplace_134StaticRegexReplace2text_vectorization/StaticRegexReplace_133:output:0*#
_output_shapes
:?????????*
pattern	 between *
rewrite ?
)text_vectorization/StaticRegexReplace_135StaticRegexReplace2text_vectorization/StaticRegexReplace_134:output:0*#
_output_shapes
:?????????*
pattern should've *
rewrite ?
)text_vectorization/StaticRegexReplace_136StaticRegexReplace2text_vectorization/StaticRegexReplace_135:output:0*#
_output_shapes
:?????????*
pattern
 hadn't *
rewrite ?
)text_vectorization/StaticRegexReplace_137StaticRegexReplace2text_vectorization/StaticRegexReplace_136:output:0*#
_output_shapes
:?????????*
pattern hasn *
rewrite ?
)text_vectorization/StaticRegexReplace_138StaticRegexReplace2text_vectorization/StaticRegexReplace_137:output:0*#
_output_shapes
:?????????*
pattern were *
rewrite ?
)text_vectorization/StaticRegexReplace_139StaticRegexReplace2text_vectorization/StaticRegexReplace_138:output:0*#
_output_shapes
:?????????*
pattern has *
rewrite ?
)text_vectorization/StaticRegexReplace_140StaticRegexReplace2text_vectorization/StaticRegexReplace_139:output:0*#
_output_shapes
:?????????*
pattern only *
rewrite ?
)text_vectorization/StaticRegexReplace_141StaticRegexReplace2text_vectorization/StaticRegexReplace_140:output:0*#
_output_shapes
:?????????*
pattern she *
rewrite ?
)text_vectorization/StaticRegexReplace_142StaticRegexReplace2text_vectorization/StaticRegexReplace_141:output:0*#
_output_shapes
:?????????*
pattern	 needn *
rewrite ?
)text_vectorization/StaticRegexReplace_143StaticRegexReplace2text_vectorization/StaticRegexReplace_142:output:0*#
_output_shapes
:?????????*
pattern	 other *
rewrite ?
)text_vectorization/StaticRegexReplace_144StaticRegexReplace2text_vectorization/StaticRegexReplace_143:output:0*#
_output_shapes
:?????????*
pattern
 hasn't *
rewrite ?
)text_vectorization/StaticRegexReplace_145StaticRegexReplace2text_vectorization/StaticRegexReplace_144:output:0*#
_output_shapes
:?????????*
pattern a *
rewrite ?
)text_vectorization/StaticRegexReplace_146StaticRegexReplace2text_vectorization/StaticRegexReplace_145:output:0*#
_output_shapes
:?????????*
pattern	 shouldn *
rewrite ?
)text_vectorization/StaticRegexReplace_147StaticRegexReplace2text_vectorization/StaticRegexReplace_146:output:0*#
_output_shapes
:?????????*
pattern and *
rewrite ?
)text_vectorization/StaticRegexReplace_148StaticRegexReplace2text_vectorization/StaticRegexReplace_147:output:0*#
_output_shapes
:?????????*
pattern	 those *
rewrite ?
)text_vectorization/StaticRegexReplace_149StaticRegexReplace2text_vectorization/StaticRegexReplace_148:output:0*#
_output_shapes
:?????????*
pattern	 being *
rewrite ?
)text_vectorization/StaticRegexReplace_150StaticRegexReplace2text_vectorization/StaticRegexReplace_149:output:0*#
_output_shapes
:?????????*
pattern such *
rewrite ?
)text_vectorization/StaticRegexReplace_151StaticRegexReplace2text_vectorization/StaticRegexReplace_150:output:0*#
_output_shapes
:?????????*
pattern as *
rewrite ?
)text_vectorization/StaticRegexReplace_152StaticRegexReplace2text_vectorization/StaticRegexReplace_151:output:0*#
_output_shapes
:?????????*
pattern ve *
rewrite ?
)text_vectorization/StaticRegexReplace_153StaticRegexReplace2text_vectorization/StaticRegexReplace_152:output:0*#
_output_shapes
:?????????*
pattern hers *
rewrite ?
)text_vectorization/StaticRegexReplace_154StaticRegexReplace2text_vectorization/StaticRegexReplace_153:output:0*#
_output_shapes
:?????????*
pattern s *
rewrite ?
)text_vectorization/StaticRegexReplace_155StaticRegexReplace2text_vectorization/StaticRegexReplace_154:output:0*#
_output_shapes
:?????????*
pattern	 their *
rewrite ?
)text_vectorization/StaticRegexReplace_156StaticRegexReplace2text_vectorization/StaticRegexReplace_155:output:0*#
_output_shapes
:?????????*
pattern	 haven't *
rewrite ?
)text_vectorization/StaticRegexReplace_157StaticRegexReplace2text_vectorization/StaticRegexReplace_156:output:0*#
_output_shapes
:?????????*
pattern for *
rewrite ?
)text_vectorization/StaticRegexReplace_158StaticRegexReplace2text_vectorization/StaticRegexReplace_157:output:0*#
_output_shapes
:?????????*
pattern if *
rewrite ?
)text_vectorization/StaticRegexReplace_159StaticRegexReplace2text_vectorization/StaticRegexReplace_158:output:0*#
_output_shapes
:?????????*
pattern that *
rewrite ?
)text_vectorization/StaticRegexReplace_160StaticRegexReplace2text_vectorization/StaticRegexReplace_159:output:0*#
_output_shapes
:?????????*
pattern isn *
rewrite ?
)text_vectorization/StaticRegexReplace_161StaticRegexReplace2text_vectorization/StaticRegexReplace_160:output:0*#
_output_shapes
:?????????*
pattern him *
rewrite ?
)text_vectorization/StaticRegexReplace_162StaticRegexReplace2text_vectorization/StaticRegexReplace_161:output:0*#
_output_shapes
:?????????*
pattern wasn *
rewrite ?
)text_vectorization/StaticRegexReplace_163StaticRegexReplace2text_vectorization/StaticRegexReplace_162:output:0*#
_output_shapes
:?????????*
pattern any *
rewrite ?
)text_vectorization/StaticRegexReplace_164StaticRegexReplace2text_vectorization/StaticRegexReplace_163:output:0*#
_output_shapes
:?????????*
pattern have *
rewrite ?
)text_vectorization/StaticRegexReplace_165StaticRegexReplace2text_vectorization/StaticRegexReplace_164:output:0*#
_output_shapes
:?????????*
pattern	 under *
rewrite ?
)text_vectorization/StaticRegexReplace_166StaticRegexReplace2text_vectorization/StaticRegexReplace_165:output:0*#
_output_shapes
:?????????*
pattern	 that'll *
rewrite ?
)text_vectorization/StaticRegexReplace_167StaticRegexReplace2text_vectorization/StaticRegexReplace_166:output:0*#
_output_shapes
:?????????*
pattern or *
rewrite ?
)text_vectorization/StaticRegexReplace_168StaticRegexReplace2text_vectorization/StaticRegexReplace_167:output:0*#
_output_shapes
:?????????*
pattern no *
rewrite ?
)text_vectorization/StaticRegexReplace_169StaticRegexReplace2text_vectorization/StaticRegexReplace_168:output:0*#
_output_shapes
:?????????*
pattern he *
rewrite ?
)text_vectorization/StaticRegexReplace_170StaticRegexReplace2text_vectorization/StaticRegexReplace_169:output:0*#
_output_shapes
:?????????*
pattern
 you're *
rewrite ?
)text_vectorization/StaticRegexReplace_171StaticRegexReplace2text_vectorization/StaticRegexReplace_170:output:0*#
_output_shapes
:?????????*
pattern this *
rewrite ?
)text_vectorization/StaticRegexReplace_172StaticRegexReplace2text_vectorization/StaticRegexReplace_171:output:0*#
_output_shapes
:?????????*
pattern	 doesn *
rewrite ?
)text_vectorization/StaticRegexReplace_173StaticRegexReplace2text_vectorization/StaticRegexReplace_172:output:0*#
_output_shapes
:?????????*
pattern	 you'd *
rewrite ?
)text_vectorization/StaticRegexReplace_174StaticRegexReplace2text_vectorization/StaticRegexReplace_173:output:0*#
_output_shapes
:?????????*
pattern up *
rewrite ?
)text_vectorization/StaticRegexReplace_175StaticRegexReplace2text_vectorization/StaticRegexReplace_174:output:0*#
_output_shapes
:?????????*
pattern
 you've *
rewrite ?
)text_vectorization/StaticRegexReplace_176StaticRegexReplace2text_vectorization/StaticRegexReplace_175:output:0*#
_output_shapes
:?????????*
pattern your *
rewrite ?
)text_vectorization/StaticRegexReplace_177StaticRegexReplace2text_vectorization/StaticRegexReplace_176:output:0*#
_output_shapes
:?????????*
pattern at *
rewrite ?
)text_vectorization/StaticRegexReplace_178StaticRegexReplace2text_vectorization/StaticRegexReplace_177:output:0*#
_output_shapes
:?????????*
pattern few *
rewrite ?
)text_vectorization/StaticRegexReplace_179StaticRegexReplace2text_vectorization/StaticRegexReplace_178:output:0*#
_output_shapes
:?????????*
pattern its *
rewrite ?
)text_vectorization/StaticRegexReplace_180StaticRegexReplace2text_vectorization/StaticRegexReplace_179:output:0*#
_output_shapes
:?????????*
pattern y *
rewrite ?
)text_vectorization/StaticRegexReplace_181StaticRegexReplace2text_vectorization/StaticRegexReplace_180:output:0*#
_output_shapes
:?????????*
pattern down *
rewrite ?
)text_vectorization/StaticRegexReplace_182StaticRegexReplace2text_vectorization/StaticRegexReplace_181:output:0*#
_output_shapes
:?????????*A
pattern64[!"\#\$%\&'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}\~]*
rewrite e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV22text_vectorization/StaticRegexReplace_182:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tGtext_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????d       ?
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????d*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
!embedding/StatefulPartitionedCallStatefulPartitionedCall?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0embedding_78221*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_77846?
(global_average_pooling1d/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_77598?
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_78225dense_78227*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_77862?
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_77943?
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_78231dense_1_78233*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_77885w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^embedding/StatefulPartitionedCall?^text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2?
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
~
)__inference_embedding_layer_call_fn_79605

inputs	
unknown:	?N 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_77846s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????d `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????d: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
٫
?
 __inference__wrapped_model_77588
text_vectorization_inputZ
Vsequential_text_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle[
Wsequential_text_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	7
3sequential_text_vectorization_string_lookup_equal_y:
6sequential_text_vectorization_string_lookup_selectv2_t	>
+sequential_embedding_embedding_lookup_77566:	?N A
/sequential_dense_matmul_readvariableop_resource:  >
0sequential_dense_biasadd_readvariableop_resource: C
1sequential_dense_1_matmul_readvariableop_resource: @
2sequential_dense_1_biasadd_readvariableop_resource:
identity??'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?(sequential/dense_1/MatMul/ReadVariableOp?%sequential/embedding/embedding_lookup?Isequential/text_vectorization/string_lookup/None_Lookup/LookupTableFindV2w
)sequential/text_vectorization/StringLowerStringLowertext_vectorization_input*#
_output_shapes
:??????????
0sequential/text_vectorization/StaticRegexReplaceStaticRegexReplace2sequential/text_vectorization/StringLower:output:0*#
_output_shapes
:?????????*
pattern<br />*
rewrite ?
2sequential/text_vectorization/StaticRegexReplace_1StaticRegexReplace9sequential/text_vectorization/StaticRegexReplace:output:0*#
_output_shapes
:?????????*+
pattern \d+(?:\.\d*)?(?:[eE][+-]?\d+)?*
rewrite ?
2sequential/text_vectorization/StaticRegexReplace_2StaticRegexReplace;sequential/text_vectorization/StaticRegexReplace_1:output:0*#
_output_shapes
:?????????*
pattern@([A-Za-z0-9_]+)*
rewrite ?
2sequential/text_vectorization/StaticRegexReplace_3StaticRegexReplace;sequential/text_vectorization/StaticRegexReplace_2:output:0*#
_output_shapes
:?????????*
pattern	 which *
rewrite ?
2sequential/text_vectorization/StaticRegexReplace_4StaticRegexReplace;sequential/text_vectorization/StaticRegexReplace_3:output:0*#
_output_shapes
:?????????*
pattern
 couldn *
rewrite ?
2sequential/text_vectorization/StaticRegexReplace_5StaticRegexReplace;sequential/text_vectorization/StaticRegexReplace_4:output:0*#
_output_shapes
:?????????*
pattern once *
rewrite ?
2sequential/text_vectorization/StaticRegexReplace_6StaticRegexReplace;sequential/text_vectorization/StaticRegexReplace_5:output:0*#
_output_shapes
:?????????*
pattern is *
rewrite ?
2sequential/text_vectorization/StaticRegexReplace_7StaticRegexReplace;sequential/text_vectorization/StaticRegexReplace_6:output:0*#
_output_shapes
:?????????*
pattern on *
rewrite ?
2sequential/text_vectorization/StaticRegexReplace_8StaticRegexReplace;sequential/text_vectorization/StaticRegexReplace_7:output:0*#
_output_shapes
:?????????*
pattern some *
rewrite ?
2sequential/text_vectorization/StaticRegexReplace_9StaticRegexReplace;sequential/text_vectorization/StaticRegexReplace_8:output:0*#
_output_shapes
:?????????*
pattern not *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_10StaticRegexReplace;sequential/text_vectorization/StaticRegexReplace_9:output:0*#
_output_shapes
:?????????*
pattern won *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_11StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_10:output:0*#
_output_shapes
:?????????*
pattern	 while *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_12StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_11:output:0*#
_output_shapes
:?????????*
pattern them *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_13StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_12:output:0*#
_output_shapes
:?????????*
pattern am *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_14StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_13:output:0*#
_output_shapes
:?????????*
pattern	 where *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_15StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_14:output:0*#
_output_shapes
:?????????*
pattern my *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_16StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_15:output:0*#
_output_shapes
:?????????*
pattern me *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_17StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_16:output:0*#
_output_shapes
:?????????*
pattern
 couldn't *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_18StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_17:output:0*#
_output_shapes
:?????????*
pattern all *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_19StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_18:output:0*#
_output_shapes
:?????????*
pattern it's *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_20StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_19:output:0*#
_output_shapes
:?????????*
pattern off *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_21StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_20:output:0*#
_output_shapes
:?????????*
pattern so *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_22StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_21:output:0*#
_output_shapes
:?????????*
pattern
 mightn *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_23StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_22:output:0*#
_output_shapes
:?????????*
pattern our *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_24StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_23:output:0*#
_output_shapes
:?????????*
pattern aren *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_25StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_24:output:0*#
_output_shapes
:?????????*
pattern	 won't *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_26StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_25:output:0*#
_output_shapes
:?????????*
pattern the *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_27StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_26:output:0*#
_output_shapes
:?????????*
pattern
 wasn't *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_28StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_27:output:0*#
_output_shapes
:?????????*
pattern just *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_29StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_28:output:0*#
_output_shapes
:?????????*
pattern
 myself *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_30StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_29:output:0*#
_output_shapes
:?????????*
pattern	 after *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_31StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_30:output:0*#
_output_shapes
:?????????*
pattern from *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_32StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_31:output:0*#
_output_shapes
:?????????*
pattern d *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_33StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_32:output:0*#
_output_shapes
:?????????*
pattern	 mustn *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_34StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_33:output:0*#
_output_shapes
:?????????*
pattern	 doesn't *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_35StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_34:output:0*#
_output_shapes
:?????????*
pattern did *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_36StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_35:output:0*#
_output_shapes
:?????????*
pattern what *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_37StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_36:output:0*#
_output_shapes
:?????????*
pattern in *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_38StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_37:output:0*#
_output_shapes
:?????????*
pattern out *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_39StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_38:output:0*#
_output_shapes
:?????????*
pattern than *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_40StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_39:output:0*#
_output_shapes
:?????????*
pattern to *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_41StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_40:output:0*#
_output_shapes
:?????????*
pattern	 because *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_42StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_41:output:0*#
_output_shapes
:?????????*
pattern too *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_43StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_42:output:0*#
_output_shapes
:?????????*
pattern here *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_44StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_43:output:0*#
_output_shapes
:?????????*
pattern ma *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_45StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_44:output:0*#
_output_shapes
:?????????*
pattern but *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_46StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_45:output:0*#
_output_shapes
:?????????*
pattern
 before *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_47StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_46:output:0*#
_output_shapes
:?????????*
pattern then *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_48StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_47:output:0*#
_output_shapes
:?????????*
pattern
 should *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_49StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_48:output:0*#
_output_shapes
:?????????*
pattern are *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_50StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_49:output:0*#
_output_shapes
:?????????*
pattern had *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_51StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_50:output:0*#
_output_shapes
:?????????*
pattern	 himself *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_52StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_51:output:0*#
_output_shapes
:?????????*
pattern you *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_53StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_52:output:0*#
_output_shapes
:?????????*
pattern
 yourself *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_54StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_53:output:0*#
_output_shapes
:?????????*
pattern	 through *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_55StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_54:output:0*#
_output_shapes
:?????????*
pattern hadn *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_56StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_55:output:0*#
_output_shapes
:?????????*
pattern does *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_57StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_56:output:0*#
_output_shapes
:?????????*
pattern m *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_58StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_57:output:0*#
_output_shapes
:?????????*
pattern ain *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_59StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_58:output:0*#
_output_shapes
:?????????*
pattern very *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_60StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_59:output:0*#
_output_shapes
:?????????*
pattern	 weren't *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_61StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_60:output:0*#
_output_shapes
:?????????*
pattern been *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_62StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_61:output:0*#
_output_shapes
:?????????*
pattern will *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_63StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_62:output:0*#
_output_shapes
:?????????*
pattern now *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_64StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_63:output:0*#
_output_shapes
:?????????*
pattern they *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_65StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_64:output:0*#
_output_shapes
:?????????*
pattern when *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_66StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_65:output:0*#
_output_shapes
:?????????*
pattern was *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_67StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_66:output:0*#
_output_shapes
:?????????*
pattern shouldn't *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_68StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_67:output:0*#
_output_shapes
:?????????*
pattern	 herself *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_69StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_68:output:0*#
_output_shapes
:?????????*
pattern	 above *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_70StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_69:output:0*#
_output_shapes
:?????????*
pattern why *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_71StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_70:output:0*#
_output_shapes
:?????????*
pattern her *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_72StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_71:output:0*#
_output_shapes
:?????????*
pattern same *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_73StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_72:output:0*#
_output_shapes
:?????????*
pattern
 having *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_74StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_73:output:0*#
_output_shapes
:?????????*
pattern	 yours *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_75StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_74:output:0*#
_output_shapes
:?????????*
pattern can *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_76StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_75:output:0*#
_output_shapes
:?????????*
pattern
 wouldn't *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_77StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_76:output:0*#
_output_shapes
:?????????*
pattern	 again *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_78StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_77:output:0*#
_output_shapes
:?????????*
pattern do *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_79StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_78:output:0*#
_output_shapes
:?????????*
pattern shan *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_80StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_79:output:0*#
_output_shapes
:?????????*
pattern	 she's *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_81StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_80:output:0*#
_output_shapes
:?????????*
pattern of *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_82StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_81:output:0*#
_output_shapes
:?????????*
pattern	 against *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_83StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_82:output:0*#
_output_shapes
:?????????*
pattern most *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_84StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_83:output:0*#
_output_shapes
:?????????*
pattern	 isn't *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_85StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_84:output:0*#
_output_shapes
:?????????*
pattern	 until *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_86StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_85:output:0*#
_output_shapes
:?????????*
pattern it *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_87StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_86:output:0*#
_output_shapes
:?????????*
pattern	 below *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_88StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_87:output:0*#
_output_shapes
:?????????*
pattern	 mustn't *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_89StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_88:output:0*#
_output_shapes
:?????????*
pattern by *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_90StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_89:output:0*#
_output_shapes
:?????????*
pattern didn *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_91StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_90:output:0*#
_output_shapes
:?????????*
pattern
 shan't *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_92StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_91:output:0*#
_output_shapes
:?????????*
pattern who *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_93StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_92:output:0*#
_output_shapes
:?????????*
pattern both *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_94StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_93:output:0*#
_output_shapes
:?????????*
pattern re *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_95StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_94:output:0*#
_output_shapes
:?????????*
pattern
 wouldn *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_96StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_95:output:0*#
_output_shapes
:?????????*
pattern his *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_97StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_96:output:0*#
_output_shapes
:?????????*
pattern ours *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_98StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_97:output:0*#
_output_shapes
:?????????*
pattern
 itself *
rewrite ?
3sequential/text_vectorization/StaticRegexReplace_99StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_98:output:0*#
_output_shapes
:?????????*
pattern don *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_100StaticRegexReplace<sequential/text_vectorization/StaticRegexReplace_99:output:0*#
_output_shapes
:?????????*
pattern	 about *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_101StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_100:output:0*#
_output_shapes
:?????????*
pattern o *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_102StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_101:output:0*#
_output_shapes
:?????????*
pattern
 during *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_103StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_102:output:0*#
_output_shapes
:?????????*
pattern whom *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_104StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_103:output:0*#
_output_shapes
:?????????*
pattern
 mightn't *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_105StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_104:output:0*#
_output_shapes
:?????????*
pattern
 didn't *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_106StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_105:output:0*#
_output_shapes
:?????????*
pattern themselves *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_107StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_106:output:0*#
_output_shapes
:?????????*
pattern with *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_108StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_107:output:0*#
_output_shapes
:?????????*
pattern
 theirs *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_109StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_108:output:0*#
_output_shapes
:?????????*
pattern	 further *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_110StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_109:output:0*#
_output_shapes
:?????????*
pattern be *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_111StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_110:output:0*#
_output_shapes
:?????????*
pattern	 weren *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_112StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_111:output:0*#
_output_shapes
:?????????*
pattern own *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_113StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_112:output:0*#
_output_shapes
:?????????*
pattern into *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_114StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_113:output:0*#
_output_shapes
:?????????*
pattern t *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_115StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_114:output:0*#
_output_shapes
:?????????*
pattern	 haven *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_116StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_115:output:0*#
_output_shapes
:?????????*
pattern	 there *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_117StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_116:output:0*#
_output_shapes
:?????????*
pattern yourselves *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_118StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_117:output:0*#
_output_shapes
:?????????*
pattern
 aren't *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_119StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_118:output:0*#
_output_shapes
:?????????*
pattern
 you'll *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_120StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_119:output:0*#
_output_shapes
:?????????*
pattern how *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_121StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_120:output:0*#
_output_shapes
:?????????*
pattern ourselves *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_122StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_121:output:0*#
_output_shapes
:?????????*
pattern an *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_123StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_122:output:0*#
_output_shapes
:?????????*
pattern	 don't *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_124StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_123:output:0*#
_output_shapes
:?????????*
pattern	 doing *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_125StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_124:output:0*#
_output_shapes
:?????????*
pattern more *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_126StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_125:output:0*#
_output_shapes
:?????????*
pattern each *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_127StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_126:output:0*#
_output_shapes
:?????????*
pattern we *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_128StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_127:output:0*#
_output_shapes
:?????????*
pattern	 these *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_129StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_128:output:0*#
_output_shapes
:?????????*
pattern over *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_130StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_129:output:0*#
_output_shapes
:?????????*
pattern i *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_131StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_130:output:0*#
_output_shapes
:?????????*
pattern nor *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_132StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_131:output:0*#
_output_shapes
:?????????*
pattern	 needn't *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_133StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_132:output:0*#
_output_shapes
:?????????*
pattern ll *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_134StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_133:output:0*#
_output_shapes
:?????????*
pattern	 between *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_135StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_134:output:0*#
_output_shapes
:?????????*
pattern should've *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_136StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_135:output:0*#
_output_shapes
:?????????*
pattern
 hadn't *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_137StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_136:output:0*#
_output_shapes
:?????????*
pattern hasn *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_138StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_137:output:0*#
_output_shapes
:?????????*
pattern were *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_139StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_138:output:0*#
_output_shapes
:?????????*
pattern has *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_140StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_139:output:0*#
_output_shapes
:?????????*
pattern only *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_141StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_140:output:0*#
_output_shapes
:?????????*
pattern she *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_142StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_141:output:0*#
_output_shapes
:?????????*
pattern	 needn *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_143StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_142:output:0*#
_output_shapes
:?????????*
pattern	 other *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_144StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_143:output:0*#
_output_shapes
:?????????*
pattern
 hasn't *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_145StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_144:output:0*#
_output_shapes
:?????????*
pattern a *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_146StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_145:output:0*#
_output_shapes
:?????????*
pattern	 shouldn *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_147StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_146:output:0*#
_output_shapes
:?????????*
pattern and *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_148StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_147:output:0*#
_output_shapes
:?????????*
pattern	 those *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_149StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_148:output:0*#
_output_shapes
:?????????*
pattern	 being *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_150StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_149:output:0*#
_output_shapes
:?????????*
pattern such *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_151StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_150:output:0*#
_output_shapes
:?????????*
pattern as *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_152StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_151:output:0*#
_output_shapes
:?????????*
pattern ve *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_153StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_152:output:0*#
_output_shapes
:?????????*
pattern hers *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_154StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_153:output:0*#
_output_shapes
:?????????*
pattern s *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_155StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_154:output:0*#
_output_shapes
:?????????*
pattern	 their *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_156StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_155:output:0*#
_output_shapes
:?????????*
pattern	 haven't *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_157StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_156:output:0*#
_output_shapes
:?????????*
pattern for *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_158StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_157:output:0*#
_output_shapes
:?????????*
pattern if *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_159StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_158:output:0*#
_output_shapes
:?????????*
pattern that *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_160StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_159:output:0*#
_output_shapes
:?????????*
pattern isn *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_161StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_160:output:0*#
_output_shapes
:?????????*
pattern him *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_162StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_161:output:0*#
_output_shapes
:?????????*
pattern wasn *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_163StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_162:output:0*#
_output_shapes
:?????????*
pattern any *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_164StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_163:output:0*#
_output_shapes
:?????????*
pattern have *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_165StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_164:output:0*#
_output_shapes
:?????????*
pattern	 under *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_166StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_165:output:0*#
_output_shapes
:?????????*
pattern	 that'll *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_167StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_166:output:0*#
_output_shapes
:?????????*
pattern or *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_168StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_167:output:0*#
_output_shapes
:?????????*
pattern no *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_169StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_168:output:0*#
_output_shapes
:?????????*
pattern he *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_170StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_169:output:0*#
_output_shapes
:?????????*
pattern
 you're *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_171StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_170:output:0*#
_output_shapes
:?????????*
pattern this *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_172StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_171:output:0*#
_output_shapes
:?????????*
pattern	 doesn *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_173StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_172:output:0*#
_output_shapes
:?????????*
pattern	 you'd *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_174StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_173:output:0*#
_output_shapes
:?????????*
pattern up *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_175StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_174:output:0*#
_output_shapes
:?????????*
pattern
 you've *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_176StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_175:output:0*#
_output_shapes
:?????????*
pattern your *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_177StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_176:output:0*#
_output_shapes
:?????????*
pattern at *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_178StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_177:output:0*#
_output_shapes
:?????????*
pattern few *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_179StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_178:output:0*#
_output_shapes
:?????????*
pattern its *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_180StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_179:output:0*#
_output_shapes
:?????????*
pattern y *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_181StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_180:output:0*#
_output_shapes
:?????????*
pattern down *
rewrite ?
4sequential/text_vectorization/StaticRegexReplace_182StaticRegexReplace=sequential/text_vectorization/StaticRegexReplace_181:output:0*#
_output_shapes
:?????????*A
pattern64[!"\#\$%\&'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}\~]*
rewrite p
/sequential/text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
7sequential/text_vectorization/StringSplit/StringSplitV2StringSplitV2=sequential/text_vectorization/StaticRegexReplace_182:output:08sequential/text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
=sequential/text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
?sequential/text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
?sequential/text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
7sequential/text_vectorization/StringSplit/strided_sliceStridedSliceAsequential/text_vectorization/StringSplit/StringSplitV2:indices:0Fsequential/text_vectorization/StringSplit/strided_slice/stack:output:0Hsequential/text_vectorization/StringSplit/strided_slice/stack_1:output:0Hsequential/text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
?sequential/text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Asequential/text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Asequential/text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
9sequential/text_vectorization/StringSplit/strided_slice_1StridedSlice?sequential/text_vectorization/StringSplit/StringSplitV2:shape:0Hsequential/text_vectorization/StringSplit/strided_slice_1/stack:output:0Jsequential/text_vectorization/StringSplit/strided_slice_1/stack_1:output:0Jsequential/text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
`sequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast@sequential/text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
bsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1CastBsequential/text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
jsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapedsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
jsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
isequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdssequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0ssequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
nsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
lsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterrsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0wsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
isequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastpsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
lsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
hsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxdsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0usequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
jsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
hsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2qsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0ssequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
hsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulmsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0lsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
lsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumfsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0lsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
lsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumfsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0psequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
lsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
msequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountdsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0psequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0usequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
gsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
bsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumtsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0psequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
ksequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
gsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
bsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2tsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0hsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0psequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Isequential/text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Vsequential_text_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle@sequential/text_vectorization/StringSplit/StringSplitV2:values:0Wsequential_text_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
1sequential/text_vectorization/string_lookup/EqualEqual@sequential/text_vectorization/StringSplit/StringSplitV2:values:03sequential_text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
4sequential/text_vectorization/string_lookup/SelectV2SelectV25sequential/text_vectorization/string_lookup/Equal:z:06sequential_text_vectorization_string_lookup_selectv2_tRsequential/text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
4sequential/text_vectorization/string_lookup/IdentityIdentity=sequential/text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????|
:sequential/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
2sequential/text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????d       ?
Asequential/text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor;sequential/text_vectorization/RaggedToTensor/Const:output:0=sequential/text_vectorization/string_lookup/Identity:output:0Csequential/text_vectorization/RaggedToTensor/default_value:output:0Bsequential/text_vectorization/StringSplit/strided_slice_1:output:0@sequential/text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????d*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
%sequential/embedding/embedding_lookupResourceGather+sequential_embedding_embedding_lookup_77566Jsequential/text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*>
_class4
20loc:@sequential/embedding/embedding_lookup/77566*+
_output_shapes
:?????????d *
dtype0?
.sequential/embedding/embedding_lookup/IdentityIdentity.sequential/embedding/embedding_lookup:output:0*
T0*>
_class4
20loc:@sequential/embedding/embedding_lookup/77566*+
_output_shapes
:?????????d ?
0sequential/embedding/embedding_lookup/Identity_1Identity7sequential/embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????d |
:sequential/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
(sequential/global_average_pooling1d/MeanMean9sequential/embedding/embedding_lookup/Identity_1:output:0Csequential/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:????????? ?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0?
sequential/dense/MatMulMatMul1sequential/global_average_pooling1d/Mean:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? r
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:????????? ~
sequential/dropout/IdentityIdentity#sequential/dense/Relu:activations:0*
T0*'
_output_shapes
:????????? ?
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
sequential/dense_1/MatMulMatMul$sequential/dropout/Identity:output:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
IdentityIdentity#sequential/dense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp&^sequential/embedding/embedding_lookupJ^sequential/text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : : : : : 2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2N
%sequential/embedding/embedding_lookup%sequential/embedding/embedding_lookup2?
Isequential/text_vectorization/string_lookup/None_Lookup/LookupTableFindV2Isequential/text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:] Y
#
_output_shapes
:?????????
2
_user_specified_nametext_vectorization_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
,
__inference__destroyer_79709
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
D__inference_embedding_layer_call_and_return_conditional_losses_79614

inputs	)
embedding_lookup_79608:	?N 
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_79608inputs*
Tindices0	*)
_class
loc:@embedding_lookup/79608*+
_output_shapes
:?????????d *
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/79608*+
_output_shapes
:?????????d ?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????d w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:?????????d Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????d: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
__inference__initializer_797047
3key_value_init3446_lookuptableimportv2_table_handle/
+key_value_init3446_lookuptableimportv2_keys1
-key_value_init3446_lookuptableimportv2_values	
identity??&key_value_init3446/LookupTableImportV2?
&key_value_init3446/LookupTableImportV2LookupTableImportV23key_value_init3446_lookuptableimportv2_table_handle+key_value_init3446_lookuptableimportv2_keys-key_value_init3446_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: o
NoOpNoOp'^key_value_init3446/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?N:?N2P
&key_value_init3446/LookupTableImportV2&key_value_init3446/LookupTableImportV2:!

_output_shapes	
:?N:!

_output_shapes	
:?N
?

?
@__inference_dense_layer_call_and_return_conditional_losses_79645

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
`
'__inference_dropout_layer_call_fn_79655

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_77943o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:????????? 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_77873

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:????????? [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:????????? "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
__inference_restore_fn_79751
restored_tensors_0
restored_tensors_1	C
?mutablehashtable_table_restore_lookuptableimportv2_table_handle
identity??2MutableHashTable_table_restore/LookupTableImportV2?
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2?mutablehashtable_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

::: 2h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:L H

_output_shapes
:
,
_user_specified_namerestored_tensors_0:LH

_output_shapes
:
,
_user_specified_namerestored_tensors_1
Ņ
?
E__inference_sequential_layer_call_and_return_conditional_losses_79598

inputsO
Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	3
 embedding_embedding_lookup_79569:	?N 6
$dense_matmul_readvariableop_resource:  3
%dense_biasadd_readvariableop_resource: 8
&dense_1_matmul_readvariableop_resource: 5
'dense_1_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?embedding/embedding_lookup?>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2Z
text_vectorization/StringLowerStringLowerinputs*#
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*#
_output_shapes
:?????????*
pattern<br />*
rewrite ?
'text_vectorization/StaticRegexReplace_1StaticRegexReplace.text_vectorization/StaticRegexReplace:output:0*#
_output_shapes
:?????????*+
pattern \d+(?:\.\d*)?(?:[eE][+-]?\d+)?*
rewrite ?
'text_vectorization/StaticRegexReplace_2StaticRegexReplace0text_vectorization/StaticRegexReplace_1:output:0*#
_output_shapes
:?????????*
pattern@([A-Za-z0-9_]+)*
rewrite ?
'text_vectorization/StaticRegexReplace_3StaticRegexReplace0text_vectorization/StaticRegexReplace_2:output:0*#
_output_shapes
:?????????*
pattern	 which *
rewrite ?
'text_vectorization/StaticRegexReplace_4StaticRegexReplace0text_vectorization/StaticRegexReplace_3:output:0*#
_output_shapes
:?????????*
pattern
 couldn *
rewrite ?
'text_vectorization/StaticRegexReplace_5StaticRegexReplace0text_vectorization/StaticRegexReplace_4:output:0*#
_output_shapes
:?????????*
pattern once *
rewrite ?
'text_vectorization/StaticRegexReplace_6StaticRegexReplace0text_vectorization/StaticRegexReplace_5:output:0*#
_output_shapes
:?????????*
pattern is *
rewrite ?
'text_vectorization/StaticRegexReplace_7StaticRegexReplace0text_vectorization/StaticRegexReplace_6:output:0*#
_output_shapes
:?????????*
pattern on *
rewrite ?
'text_vectorization/StaticRegexReplace_8StaticRegexReplace0text_vectorization/StaticRegexReplace_7:output:0*#
_output_shapes
:?????????*
pattern some *
rewrite ?
'text_vectorization/StaticRegexReplace_9StaticRegexReplace0text_vectorization/StaticRegexReplace_8:output:0*#
_output_shapes
:?????????*
pattern not *
rewrite ?
(text_vectorization/StaticRegexReplace_10StaticRegexReplace0text_vectorization/StaticRegexReplace_9:output:0*#
_output_shapes
:?????????*
pattern won *
rewrite ?
(text_vectorization/StaticRegexReplace_11StaticRegexReplace1text_vectorization/StaticRegexReplace_10:output:0*#
_output_shapes
:?????????*
pattern	 while *
rewrite ?
(text_vectorization/StaticRegexReplace_12StaticRegexReplace1text_vectorization/StaticRegexReplace_11:output:0*#
_output_shapes
:?????????*
pattern them *
rewrite ?
(text_vectorization/StaticRegexReplace_13StaticRegexReplace1text_vectorization/StaticRegexReplace_12:output:0*#
_output_shapes
:?????????*
pattern am *
rewrite ?
(text_vectorization/StaticRegexReplace_14StaticRegexReplace1text_vectorization/StaticRegexReplace_13:output:0*#
_output_shapes
:?????????*
pattern	 where *
rewrite ?
(text_vectorization/StaticRegexReplace_15StaticRegexReplace1text_vectorization/StaticRegexReplace_14:output:0*#
_output_shapes
:?????????*
pattern my *
rewrite ?
(text_vectorization/StaticRegexReplace_16StaticRegexReplace1text_vectorization/StaticRegexReplace_15:output:0*#
_output_shapes
:?????????*
pattern me *
rewrite ?
(text_vectorization/StaticRegexReplace_17StaticRegexReplace1text_vectorization/StaticRegexReplace_16:output:0*#
_output_shapes
:?????????*
pattern
 couldn't *
rewrite ?
(text_vectorization/StaticRegexReplace_18StaticRegexReplace1text_vectorization/StaticRegexReplace_17:output:0*#
_output_shapes
:?????????*
pattern all *
rewrite ?
(text_vectorization/StaticRegexReplace_19StaticRegexReplace1text_vectorization/StaticRegexReplace_18:output:0*#
_output_shapes
:?????????*
pattern it's *
rewrite ?
(text_vectorization/StaticRegexReplace_20StaticRegexReplace1text_vectorization/StaticRegexReplace_19:output:0*#
_output_shapes
:?????????*
pattern off *
rewrite ?
(text_vectorization/StaticRegexReplace_21StaticRegexReplace1text_vectorization/StaticRegexReplace_20:output:0*#
_output_shapes
:?????????*
pattern so *
rewrite ?
(text_vectorization/StaticRegexReplace_22StaticRegexReplace1text_vectorization/StaticRegexReplace_21:output:0*#
_output_shapes
:?????????*
pattern
 mightn *
rewrite ?
(text_vectorization/StaticRegexReplace_23StaticRegexReplace1text_vectorization/StaticRegexReplace_22:output:0*#
_output_shapes
:?????????*
pattern our *
rewrite ?
(text_vectorization/StaticRegexReplace_24StaticRegexReplace1text_vectorization/StaticRegexReplace_23:output:0*#
_output_shapes
:?????????*
pattern aren *
rewrite ?
(text_vectorization/StaticRegexReplace_25StaticRegexReplace1text_vectorization/StaticRegexReplace_24:output:0*#
_output_shapes
:?????????*
pattern	 won't *
rewrite ?
(text_vectorization/StaticRegexReplace_26StaticRegexReplace1text_vectorization/StaticRegexReplace_25:output:0*#
_output_shapes
:?????????*
pattern the *
rewrite ?
(text_vectorization/StaticRegexReplace_27StaticRegexReplace1text_vectorization/StaticRegexReplace_26:output:0*#
_output_shapes
:?????????*
pattern
 wasn't *
rewrite ?
(text_vectorization/StaticRegexReplace_28StaticRegexReplace1text_vectorization/StaticRegexReplace_27:output:0*#
_output_shapes
:?????????*
pattern just *
rewrite ?
(text_vectorization/StaticRegexReplace_29StaticRegexReplace1text_vectorization/StaticRegexReplace_28:output:0*#
_output_shapes
:?????????*
pattern
 myself *
rewrite ?
(text_vectorization/StaticRegexReplace_30StaticRegexReplace1text_vectorization/StaticRegexReplace_29:output:0*#
_output_shapes
:?????????*
pattern	 after *
rewrite ?
(text_vectorization/StaticRegexReplace_31StaticRegexReplace1text_vectorization/StaticRegexReplace_30:output:0*#
_output_shapes
:?????????*
pattern from *
rewrite ?
(text_vectorization/StaticRegexReplace_32StaticRegexReplace1text_vectorization/StaticRegexReplace_31:output:0*#
_output_shapes
:?????????*
pattern d *
rewrite ?
(text_vectorization/StaticRegexReplace_33StaticRegexReplace1text_vectorization/StaticRegexReplace_32:output:0*#
_output_shapes
:?????????*
pattern	 mustn *
rewrite ?
(text_vectorization/StaticRegexReplace_34StaticRegexReplace1text_vectorization/StaticRegexReplace_33:output:0*#
_output_shapes
:?????????*
pattern	 doesn't *
rewrite ?
(text_vectorization/StaticRegexReplace_35StaticRegexReplace1text_vectorization/StaticRegexReplace_34:output:0*#
_output_shapes
:?????????*
pattern did *
rewrite ?
(text_vectorization/StaticRegexReplace_36StaticRegexReplace1text_vectorization/StaticRegexReplace_35:output:0*#
_output_shapes
:?????????*
pattern what *
rewrite ?
(text_vectorization/StaticRegexReplace_37StaticRegexReplace1text_vectorization/StaticRegexReplace_36:output:0*#
_output_shapes
:?????????*
pattern in *
rewrite ?
(text_vectorization/StaticRegexReplace_38StaticRegexReplace1text_vectorization/StaticRegexReplace_37:output:0*#
_output_shapes
:?????????*
pattern out *
rewrite ?
(text_vectorization/StaticRegexReplace_39StaticRegexReplace1text_vectorization/StaticRegexReplace_38:output:0*#
_output_shapes
:?????????*
pattern than *
rewrite ?
(text_vectorization/StaticRegexReplace_40StaticRegexReplace1text_vectorization/StaticRegexReplace_39:output:0*#
_output_shapes
:?????????*
pattern to *
rewrite ?
(text_vectorization/StaticRegexReplace_41StaticRegexReplace1text_vectorization/StaticRegexReplace_40:output:0*#
_output_shapes
:?????????*
pattern	 because *
rewrite ?
(text_vectorization/StaticRegexReplace_42StaticRegexReplace1text_vectorization/StaticRegexReplace_41:output:0*#
_output_shapes
:?????????*
pattern too *
rewrite ?
(text_vectorization/StaticRegexReplace_43StaticRegexReplace1text_vectorization/StaticRegexReplace_42:output:0*#
_output_shapes
:?????????*
pattern here *
rewrite ?
(text_vectorization/StaticRegexReplace_44StaticRegexReplace1text_vectorization/StaticRegexReplace_43:output:0*#
_output_shapes
:?????????*
pattern ma *
rewrite ?
(text_vectorization/StaticRegexReplace_45StaticRegexReplace1text_vectorization/StaticRegexReplace_44:output:0*#
_output_shapes
:?????????*
pattern but *
rewrite ?
(text_vectorization/StaticRegexReplace_46StaticRegexReplace1text_vectorization/StaticRegexReplace_45:output:0*#
_output_shapes
:?????????*
pattern
 before *
rewrite ?
(text_vectorization/StaticRegexReplace_47StaticRegexReplace1text_vectorization/StaticRegexReplace_46:output:0*#
_output_shapes
:?????????*
pattern then *
rewrite ?
(text_vectorization/StaticRegexReplace_48StaticRegexReplace1text_vectorization/StaticRegexReplace_47:output:0*#
_output_shapes
:?????????*
pattern
 should *
rewrite ?
(text_vectorization/StaticRegexReplace_49StaticRegexReplace1text_vectorization/StaticRegexReplace_48:output:0*#
_output_shapes
:?????????*
pattern are *
rewrite ?
(text_vectorization/StaticRegexReplace_50StaticRegexReplace1text_vectorization/StaticRegexReplace_49:output:0*#
_output_shapes
:?????????*
pattern had *
rewrite ?
(text_vectorization/StaticRegexReplace_51StaticRegexReplace1text_vectorization/StaticRegexReplace_50:output:0*#
_output_shapes
:?????????*
pattern	 himself *
rewrite ?
(text_vectorization/StaticRegexReplace_52StaticRegexReplace1text_vectorization/StaticRegexReplace_51:output:0*#
_output_shapes
:?????????*
pattern you *
rewrite ?
(text_vectorization/StaticRegexReplace_53StaticRegexReplace1text_vectorization/StaticRegexReplace_52:output:0*#
_output_shapes
:?????????*
pattern
 yourself *
rewrite ?
(text_vectorization/StaticRegexReplace_54StaticRegexReplace1text_vectorization/StaticRegexReplace_53:output:0*#
_output_shapes
:?????????*
pattern	 through *
rewrite ?
(text_vectorization/StaticRegexReplace_55StaticRegexReplace1text_vectorization/StaticRegexReplace_54:output:0*#
_output_shapes
:?????????*
pattern hadn *
rewrite ?
(text_vectorization/StaticRegexReplace_56StaticRegexReplace1text_vectorization/StaticRegexReplace_55:output:0*#
_output_shapes
:?????????*
pattern does *
rewrite ?
(text_vectorization/StaticRegexReplace_57StaticRegexReplace1text_vectorization/StaticRegexReplace_56:output:0*#
_output_shapes
:?????????*
pattern m *
rewrite ?
(text_vectorization/StaticRegexReplace_58StaticRegexReplace1text_vectorization/StaticRegexReplace_57:output:0*#
_output_shapes
:?????????*
pattern ain *
rewrite ?
(text_vectorization/StaticRegexReplace_59StaticRegexReplace1text_vectorization/StaticRegexReplace_58:output:0*#
_output_shapes
:?????????*
pattern very *
rewrite ?
(text_vectorization/StaticRegexReplace_60StaticRegexReplace1text_vectorization/StaticRegexReplace_59:output:0*#
_output_shapes
:?????????*
pattern	 weren't *
rewrite ?
(text_vectorization/StaticRegexReplace_61StaticRegexReplace1text_vectorization/StaticRegexReplace_60:output:0*#
_output_shapes
:?????????*
pattern been *
rewrite ?
(text_vectorization/StaticRegexReplace_62StaticRegexReplace1text_vectorization/StaticRegexReplace_61:output:0*#
_output_shapes
:?????????*
pattern will *
rewrite ?
(text_vectorization/StaticRegexReplace_63StaticRegexReplace1text_vectorization/StaticRegexReplace_62:output:0*#
_output_shapes
:?????????*
pattern now *
rewrite ?
(text_vectorization/StaticRegexReplace_64StaticRegexReplace1text_vectorization/StaticRegexReplace_63:output:0*#
_output_shapes
:?????????*
pattern they *
rewrite ?
(text_vectorization/StaticRegexReplace_65StaticRegexReplace1text_vectorization/StaticRegexReplace_64:output:0*#
_output_shapes
:?????????*
pattern when *
rewrite ?
(text_vectorization/StaticRegexReplace_66StaticRegexReplace1text_vectorization/StaticRegexReplace_65:output:0*#
_output_shapes
:?????????*
pattern was *
rewrite ?
(text_vectorization/StaticRegexReplace_67StaticRegexReplace1text_vectorization/StaticRegexReplace_66:output:0*#
_output_shapes
:?????????*
pattern shouldn't *
rewrite ?
(text_vectorization/StaticRegexReplace_68StaticRegexReplace1text_vectorization/StaticRegexReplace_67:output:0*#
_output_shapes
:?????????*
pattern	 herself *
rewrite ?
(text_vectorization/StaticRegexReplace_69StaticRegexReplace1text_vectorization/StaticRegexReplace_68:output:0*#
_output_shapes
:?????????*
pattern	 above *
rewrite ?
(text_vectorization/StaticRegexReplace_70StaticRegexReplace1text_vectorization/StaticRegexReplace_69:output:0*#
_output_shapes
:?????????*
pattern why *
rewrite ?
(text_vectorization/StaticRegexReplace_71StaticRegexReplace1text_vectorization/StaticRegexReplace_70:output:0*#
_output_shapes
:?????????*
pattern her *
rewrite ?
(text_vectorization/StaticRegexReplace_72StaticRegexReplace1text_vectorization/StaticRegexReplace_71:output:0*#
_output_shapes
:?????????*
pattern same *
rewrite ?
(text_vectorization/StaticRegexReplace_73StaticRegexReplace1text_vectorization/StaticRegexReplace_72:output:0*#
_output_shapes
:?????????*
pattern
 having *
rewrite ?
(text_vectorization/StaticRegexReplace_74StaticRegexReplace1text_vectorization/StaticRegexReplace_73:output:0*#
_output_shapes
:?????????*
pattern	 yours *
rewrite ?
(text_vectorization/StaticRegexReplace_75StaticRegexReplace1text_vectorization/StaticRegexReplace_74:output:0*#
_output_shapes
:?????????*
pattern can *
rewrite ?
(text_vectorization/StaticRegexReplace_76StaticRegexReplace1text_vectorization/StaticRegexReplace_75:output:0*#
_output_shapes
:?????????*
pattern
 wouldn't *
rewrite ?
(text_vectorization/StaticRegexReplace_77StaticRegexReplace1text_vectorization/StaticRegexReplace_76:output:0*#
_output_shapes
:?????????*
pattern	 again *
rewrite ?
(text_vectorization/StaticRegexReplace_78StaticRegexReplace1text_vectorization/StaticRegexReplace_77:output:0*#
_output_shapes
:?????????*
pattern do *
rewrite ?
(text_vectorization/StaticRegexReplace_79StaticRegexReplace1text_vectorization/StaticRegexReplace_78:output:0*#
_output_shapes
:?????????*
pattern shan *
rewrite ?
(text_vectorization/StaticRegexReplace_80StaticRegexReplace1text_vectorization/StaticRegexReplace_79:output:0*#
_output_shapes
:?????????*
pattern	 she's *
rewrite ?
(text_vectorization/StaticRegexReplace_81StaticRegexReplace1text_vectorization/StaticRegexReplace_80:output:0*#
_output_shapes
:?????????*
pattern of *
rewrite ?
(text_vectorization/StaticRegexReplace_82StaticRegexReplace1text_vectorization/StaticRegexReplace_81:output:0*#
_output_shapes
:?????????*
pattern	 against *
rewrite ?
(text_vectorization/StaticRegexReplace_83StaticRegexReplace1text_vectorization/StaticRegexReplace_82:output:0*#
_output_shapes
:?????????*
pattern most *
rewrite ?
(text_vectorization/StaticRegexReplace_84StaticRegexReplace1text_vectorization/StaticRegexReplace_83:output:0*#
_output_shapes
:?????????*
pattern	 isn't *
rewrite ?
(text_vectorization/StaticRegexReplace_85StaticRegexReplace1text_vectorization/StaticRegexReplace_84:output:0*#
_output_shapes
:?????????*
pattern	 until *
rewrite ?
(text_vectorization/StaticRegexReplace_86StaticRegexReplace1text_vectorization/StaticRegexReplace_85:output:0*#
_output_shapes
:?????????*
pattern it *
rewrite ?
(text_vectorization/StaticRegexReplace_87StaticRegexReplace1text_vectorization/StaticRegexReplace_86:output:0*#
_output_shapes
:?????????*
pattern	 below *
rewrite ?
(text_vectorization/StaticRegexReplace_88StaticRegexReplace1text_vectorization/StaticRegexReplace_87:output:0*#
_output_shapes
:?????????*
pattern	 mustn't *
rewrite ?
(text_vectorization/StaticRegexReplace_89StaticRegexReplace1text_vectorization/StaticRegexReplace_88:output:0*#
_output_shapes
:?????????*
pattern by *
rewrite ?
(text_vectorization/StaticRegexReplace_90StaticRegexReplace1text_vectorization/StaticRegexReplace_89:output:0*#
_output_shapes
:?????????*
pattern didn *
rewrite ?
(text_vectorization/StaticRegexReplace_91StaticRegexReplace1text_vectorization/StaticRegexReplace_90:output:0*#
_output_shapes
:?????????*
pattern
 shan't *
rewrite ?
(text_vectorization/StaticRegexReplace_92StaticRegexReplace1text_vectorization/StaticRegexReplace_91:output:0*#
_output_shapes
:?????????*
pattern who *
rewrite ?
(text_vectorization/StaticRegexReplace_93StaticRegexReplace1text_vectorization/StaticRegexReplace_92:output:0*#
_output_shapes
:?????????*
pattern both *
rewrite ?
(text_vectorization/StaticRegexReplace_94StaticRegexReplace1text_vectorization/StaticRegexReplace_93:output:0*#
_output_shapes
:?????????*
pattern re *
rewrite ?
(text_vectorization/StaticRegexReplace_95StaticRegexReplace1text_vectorization/StaticRegexReplace_94:output:0*#
_output_shapes
:?????????*
pattern
 wouldn *
rewrite ?
(text_vectorization/StaticRegexReplace_96StaticRegexReplace1text_vectorization/StaticRegexReplace_95:output:0*#
_output_shapes
:?????????*
pattern his *
rewrite ?
(text_vectorization/StaticRegexReplace_97StaticRegexReplace1text_vectorization/StaticRegexReplace_96:output:0*#
_output_shapes
:?????????*
pattern ours *
rewrite ?
(text_vectorization/StaticRegexReplace_98StaticRegexReplace1text_vectorization/StaticRegexReplace_97:output:0*#
_output_shapes
:?????????*
pattern
 itself *
rewrite ?
(text_vectorization/StaticRegexReplace_99StaticRegexReplace1text_vectorization/StaticRegexReplace_98:output:0*#
_output_shapes
:?????????*
pattern don *
rewrite ?
)text_vectorization/StaticRegexReplace_100StaticRegexReplace1text_vectorization/StaticRegexReplace_99:output:0*#
_output_shapes
:?????????*
pattern	 about *
rewrite ?
)text_vectorization/StaticRegexReplace_101StaticRegexReplace2text_vectorization/StaticRegexReplace_100:output:0*#
_output_shapes
:?????????*
pattern o *
rewrite ?
)text_vectorization/StaticRegexReplace_102StaticRegexReplace2text_vectorization/StaticRegexReplace_101:output:0*#
_output_shapes
:?????????*
pattern
 during *
rewrite ?
)text_vectorization/StaticRegexReplace_103StaticRegexReplace2text_vectorization/StaticRegexReplace_102:output:0*#
_output_shapes
:?????????*
pattern whom *
rewrite ?
)text_vectorization/StaticRegexReplace_104StaticRegexReplace2text_vectorization/StaticRegexReplace_103:output:0*#
_output_shapes
:?????????*
pattern
 mightn't *
rewrite ?
)text_vectorization/StaticRegexReplace_105StaticRegexReplace2text_vectorization/StaticRegexReplace_104:output:0*#
_output_shapes
:?????????*
pattern
 didn't *
rewrite ?
)text_vectorization/StaticRegexReplace_106StaticRegexReplace2text_vectorization/StaticRegexReplace_105:output:0*#
_output_shapes
:?????????*
pattern themselves *
rewrite ?
)text_vectorization/StaticRegexReplace_107StaticRegexReplace2text_vectorization/StaticRegexReplace_106:output:0*#
_output_shapes
:?????????*
pattern with *
rewrite ?
)text_vectorization/StaticRegexReplace_108StaticRegexReplace2text_vectorization/StaticRegexReplace_107:output:0*#
_output_shapes
:?????????*
pattern
 theirs *
rewrite ?
)text_vectorization/StaticRegexReplace_109StaticRegexReplace2text_vectorization/StaticRegexReplace_108:output:0*#
_output_shapes
:?????????*
pattern	 further *
rewrite ?
)text_vectorization/StaticRegexReplace_110StaticRegexReplace2text_vectorization/StaticRegexReplace_109:output:0*#
_output_shapes
:?????????*
pattern be *
rewrite ?
)text_vectorization/StaticRegexReplace_111StaticRegexReplace2text_vectorization/StaticRegexReplace_110:output:0*#
_output_shapes
:?????????*
pattern	 weren *
rewrite ?
)text_vectorization/StaticRegexReplace_112StaticRegexReplace2text_vectorization/StaticRegexReplace_111:output:0*#
_output_shapes
:?????????*
pattern own *
rewrite ?
)text_vectorization/StaticRegexReplace_113StaticRegexReplace2text_vectorization/StaticRegexReplace_112:output:0*#
_output_shapes
:?????????*
pattern into *
rewrite ?
)text_vectorization/StaticRegexReplace_114StaticRegexReplace2text_vectorization/StaticRegexReplace_113:output:0*#
_output_shapes
:?????????*
pattern t *
rewrite ?
)text_vectorization/StaticRegexReplace_115StaticRegexReplace2text_vectorization/StaticRegexReplace_114:output:0*#
_output_shapes
:?????????*
pattern	 haven *
rewrite ?
)text_vectorization/StaticRegexReplace_116StaticRegexReplace2text_vectorization/StaticRegexReplace_115:output:0*#
_output_shapes
:?????????*
pattern	 there *
rewrite ?
)text_vectorization/StaticRegexReplace_117StaticRegexReplace2text_vectorization/StaticRegexReplace_116:output:0*#
_output_shapes
:?????????*
pattern yourselves *
rewrite ?
)text_vectorization/StaticRegexReplace_118StaticRegexReplace2text_vectorization/StaticRegexReplace_117:output:0*#
_output_shapes
:?????????*
pattern
 aren't *
rewrite ?
)text_vectorization/StaticRegexReplace_119StaticRegexReplace2text_vectorization/StaticRegexReplace_118:output:0*#
_output_shapes
:?????????*
pattern
 you'll *
rewrite ?
)text_vectorization/StaticRegexReplace_120StaticRegexReplace2text_vectorization/StaticRegexReplace_119:output:0*#
_output_shapes
:?????????*
pattern how *
rewrite ?
)text_vectorization/StaticRegexReplace_121StaticRegexReplace2text_vectorization/StaticRegexReplace_120:output:0*#
_output_shapes
:?????????*
pattern ourselves *
rewrite ?
)text_vectorization/StaticRegexReplace_122StaticRegexReplace2text_vectorization/StaticRegexReplace_121:output:0*#
_output_shapes
:?????????*
pattern an *
rewrite ?
)text_vectorization/StaticRegexReplace_123StaticRegexReplace2text_vectorization/StaticRegexReplace_122:output:0*#
_output_shapes
:?????????*
pattern	 don't *
rewrite ?
)text_vectorization/StaticRegexReplace_124StaticRegexReplace2text_vectorization/StaticRegexReplace_123:output:0*#
_output_shapes
:?????????*
pattern	 doing *
rewrite ?
)text_vectorization/StaticRegexReplace_125StaticRegexReplace2text_vectorization/StaticRegexReplace_124:output:0*#
_output_shapes
:?????????*
pattern more *
rewrite ?
)text_vectorization/StaticRegexReplace_126StaticRegexReplace2text_vectorization/StaticRegexReplace_125:output:0*#
_output_shapes
:?????????*
pattern each *
rewrite ?
)text_vectorization/StaticRegexReplace_127StaticRegexReplace2text_vectorization/StaticRegexReplace_126:output:0*#
_output_shapes
:?????????*
pattern we *
rewrite ?
)text_vectorization/StaticRegexReplace_128StaticRegexReplace2text_vectorization/StaticRegexReplace_127:output:0*#
_output_shapes
:?????????*
pattern	 these *
rewrite ?
)text_vectorization/StaticRegexReplace_129StaticRegexReplace2text_vectorization/StaticRegexReplace_128:output:0*#
_output_shapes
:?????????*
pattern over *
rewrite ?
)text_vectorization/StaticRegexReplace_130StaticRegexReplace2text_vectorization/StaticRegexReplace_129:output:0*#
_output_shapes
:?????????*
pattern i *
rewrite ?
)text_vectorization/StaticRegexReplace_131StaticRegexReplace2text_vectorization/StaticRegexReplace_130:output:0*#
_output_shapes
:?????????*
pattern nor *
rewrite ?
)text_vectorization/StaticRegexReplace_132StaticRegexReplace2text_vectorization/StaticRegexReplace_131:output:0*#
_output_shapes
:?????????*
pattern	 needn't *
rewrite ?
)text_vectorization/StaticRegexReplace_133StaticRegexReplace2text_vectorization/StaticRegexReplace_132:output:0*#
_output_shapes
:?????????*
pattern ll *
rewrite ?
)text_vectorization/StaticRegexReplace_134StaticRegexReplace2text_vectorization/StaticRegexReplace_133:output:0*#
_output_shapes
:?????????*
pattern	 between *
rewrite ?
)text_vectorization/StaticRegexReplace_135StaticRegexReplace2text_vectorization/StaticRegexReplace_134:output:0*#
_output_shapes
:?????????*
pattern should've *
rewrite ?
)text_vectorization/StaticRegexReplace_136StaticRegexReplace2text_vectorization/StaticRegexReplace_135:output:0*#
_output_shapes
:?????????*
pattern
 hadn't *
rewrite ?
)text_vectorization/StaticRegexReplace_137StaticRegexReplace2text_vectorization/StaticRegexReplace_136:output:0*#
_output_shapes
:?????????*
pattern hasn *
rewrite ?
)text_vectorization/StaticRegexReplace_138StaticRegexReplace2text_vectorization/StaticRegexReplace_137:output:0*#
_output_shapes
:?????????*
pattern were *
rewrite ?
)text_vectorization/StaticRegexReplace_139StaticRegexReplace2text_vectorization/StaticRegexReplace_138:output:0*#
_output_shapes
:?????????*
pattern has *
rewrite ?
)text_vectorization/StaticRegexReplace_140StaticRegexReplace2text_vectorization/StaticRegexReplace_139:output:0*#
_output_shapes
:?????????*
pattern only *
rewrite ?
)text_vectorization/StaticRegexReplace_141StaticRegexReplace2text_vectorization/StaticRegexReplace_140:output:0*#
_output_shapes
:?????????*
pattern she *
rewrite ?
)text_vectorization/StaticRegexReplace_142StaticRegexReplace2text_vectorization/StaticRegexReplace_141:output:0*#
_output_shapes
:?????????*
pattern	 needn *
rewrite ?
)text_vectorization/StaticRegexReplace_143StaticRegexReplace2text_vectorization/StaticRegexReplace_142:output:0*#
_output_shapes
:?????????*
pattern	 other *
rewrite ?
)text_vectorization/StaticRegexReplace_144StaticRegexReplace2text_vectorization/StaticRegexReplace_143:output:0*#
_output_shapes
:?????????*
pattern
 hasn't *
rewrite ?
)text_vectorization/StaticRegexReplace_145StaticRegexReplace2text_vectorization/StaticRegexReplace_144:output:0*#
_output_shapes
:?????????*
pattern a *
rewrite ?
)text_vectorization/StaticRegexReplace_146StaticRegexReplace2text_vectorization/StaticRegexReplace_145:output:0*#
_output_shapes
:?????????*
pattern	 shouldn *
rewrite ?
)text_vectorization/StaticRegexReplace_147StaticRegexReplace2text_vectorization/StaticRegexReplace_146:output:0*#
_output_shapes
:?????????*
pattern and *
rewrite ?
)text_vectorization/StaticRegexReplace_148StaticRegexReplace2text_vectorization/StaticRegexReplace_147:output:0*#
_output_shapes
:?????????*
pattern	 those *
rewrite ?
)text_vectorization/StaticRegexReplace_149StaticRegexReplace2text_vectorization/StaticRegexReplace_148:output:0*#
_output_shapes
:?????????*
pattern	 being *
rewrite ?
)text_vectorization/StaticRegexReplace_150StaticRegexReplace2text_vectorization/StaticRegexReplace_149:output:0*#
_output_shapes
:?????????*
pattern such *
rewrite ?
)text_vectorization/StaticRegexReplace_151StaticRegexReplace2text_vectorization/StaticRegexReplace_150:output:0*#
_output_shapes
:?????????*
pattern as *
rewrite ?
)text_vectorization/StaticRegexReplace_152StaticRegexReplace2text_vectorization/StaticRegexReplace_151:output:0*#
_output_shapes
:?????????*
pattern ve *
rewrite ?
)text_vectorization/StaticRegexReplace_153StaticRegexReplace2text_vectorization/StaticRegexReplace_152:output:0*#
_output_shapes
:?????????*
pattern hers *
rewrite ?
)text_vectorization/StaticRegexReplace_154StaticRegexReplace2text_vectorization/StaticRegexReplace_153:output:0*#
_output_shapes
:?????????*
pattern s *
rewrite ?
)text_vectorization/StaticRegexReplace_155StaticRegexReplace2text_vectorization/StaticRegexReplace_154:output:0*#
_output_shapes
:?????????*
pattern	 their *
rewrite ?
)text_vectorization/StaticRegexReplace_156StaticRegexReplace2text_vectorization/StaticRegexReplace_155:output:0*#
_output_shapes
:?????????*
pattern	 haven't *
rewrite ?
)text_vectorization/StaticRegexReplace_157StaticRegexReplace2text_vectorization/StaticRegexReplace_156:output:0*#
_output_shapes
:?????????*
pattern for *
rewrite ?
)text_vectorization/StaticRegexReplace_158StaticRegexReplace2text_vectorization/StaticRegexReplace_157:output:0*#
_output_shapes
:?????????*
pattern if *
rewrite ?
)text_vectorization/StaticRegexReplace_159StaticRegexReplace2text_vectorization/StaticRegexReplace_158:output:0*#
_output_shapes
:?????????*
pattern that *
rewrite ?
)text_vectorization/StaticRegexReplace_160StaticRegexReplace2text_vectorization/StaticRegexReplace_159:output:0*#
_output_shapes
:?????????*
pattern isn *
rewrite ?
)text_vectorization/StaticRegexReplace_161StaticRegexReplace2text_vectorization/StaticRegexReplace_160:output:0*#
_output_shapes
:?????????*
pattern him *
rewrite ?
)text_vectorization/StaticRegexReplace_162StaticRegexReplace2text_vectorization/StaticRegexReplace_161:output:0*#
_output_shapes
:?????????*
pattern wasn *
rewrite ?
)text_vectorization/StaticRegexReplace_163StaticRegexReplace2text_vectorization/StaticRegexReplace_162:output:0*#
_output_shapes
:?????????*
pattern any *
rewrite ?
)text_vectorization/StaticRegexReplace_164StaticRegexReplace2text_vectorization/StaticRegexReplace_163:output:0*#
_output_shapes
:?????????*
pattern have *
rewrite ?
)text_vectorization/StaticRegexReplace_165StaticRegexReplace2text_vectorization/StaticRegexReplace_164:output:0*#
_output_shapes
:?????????*
pattern	 under *
rewrite ?
)text_vectorization/StaticRegexReplace_166StaticRegexReplace2text_vectorization/StaticRegexReplace_165:output:0*#
_output_shapes
:?????????*
pattern	 that'll *
rewrite ?
)text_vectorization/StaticRegexReplace_167StaticRegexReplace2text_vectorization/StaticRegexReplace_166:output:0*#
_output_shapes
:?????????*
pattern or *
rewrite ?
)text_vectorization/StaticRegexReplace_168StaticRegexReplace2text_vectorization/StaticRegexReplace_167:output:0*#
_output_shapes
:?????????*
pattern no *
rewrite ?
)text_vectorization/StaticRegexReplace_169StaticRegexReplace2text_vectorization/StaticRegexReplace_168:output:0*#
_output_shapes
:?????????*
pattern he *
rewrite ?
)text_vectorization/StaticRegexReplace_170StaticRegexReplace2text_vectorization/StaticRegexReplace_169:output:0*#
_output_shapes
:?????????*
pattern
 you're *
rewrite ?
)text_vectorization/StaticRegexReplace_171StaticRegexReplace2text_vectorization/StaticRegexReplace_170:output:0*#
_output_shapes
:?????????*
pattern this *
rewrite ?
)text_vectorization/StaticRegexReplace_172StaticRegexReplace2text_vectorization/StaticRegexReplace_171:output:0*#
_output_shapes
:?????????*
pattern	 doesn *
rewrite ?
)text_vectorization/StaticRegexReplace_173StaticRegexReplace2text_vectorization/StaticRegexReplace_172:output:0*#
_output_shapes
:?????????*
pattern	 you'd *
rewrite ?
)text_vectorization/StaticRegexReplace_174StaticRegexReplace2text_vectorization/StaticRegexReplace_173:output:0*#
_output_shapes
:?????????*
pattern up *
rewrite ?
)text_vectorization/StaticRegexReplace_175StaticRegexReplace2text_vectorization/StaticRegexReplace_174:output:0*#
_output_shapes
:?????????*
pattern
 you've *
rewrite ?
)text_vectorization/StaticRegexReplace_176StaticRegexReplace2text_vectorization/StaticRegexReplace_175:output:0*#
_output_shapes
:?????????*
pattern your *
rewrite ?
)text_vectorization/StaticRegexReplace_177StaticRegexReplace2text_vectorization/StaticRegexReplace_176:output:0*#
_output_shapes
:?????????*
pattern at *
rewrite ?
)text_vectorization/StaticRegexReplace_178StaticRegexReplace2text_vectorization/StaticRegexReplace_177:output:0*#
_output_shapes
:?????????*
pattern few *
rewrite ?
)text_vectorization/StaticRegexReplace_179StaticRegexReplace2text_vectorization/StaticRegexReplace_178:output:0*#
_output_shapes
:?????????*
pattern its *
rewrite ?
)text_vectorization/StaticRegexReplace_180StaticRegexReplace2text_vectorization/StaticRegexReplace_179:output:0*#
_output_shapes
:?????????*
pattern y *
rewrite ?
)text_vectorization/StaticRegexReplace_181StaticRegexReplace2text_vectorization/StaticRegexReplace_180:output:0*#
_output_shapes
:?????????*
pattern down *
rewrite ?
)text_vectorization/StaticRegexReplace_182StaticRegexReplace2text_vectorization/StaticRegexReplace_181:output:0*#
_output_shapes
:?????????*A
pattern64[!"\#\$%\&'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}\~]*
rewrite e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV22text_vectorization/StaticRegexReplace_182:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tGtext_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????d       ?
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????d*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
embedding/embedding_lookupResourceGather embedding_embedding_lookup_79569?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*3
_class)
'%loc:@embedding/embedding_lookup/79569*+
_output_shapes
:?????????d *
dtype0?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*3
_class)
'%loc:@embedding/embedding_lookup/79569*+
_output_shapes
:?????????d ?
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????d q
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
global_average_pooling1d/MeanMean.embedding/embedding_lookup/Identity_1:output:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:????????? ?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0?
dense/MatMulMatMul&global_average_pooling1d/Mean:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? \

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:????????? Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
dropout/dropout/MulMuldense/Relu:activations:0dropout/dropout/Const:output:0*
T0*'
_output_shapes
:????????? ]
dropout/dropout/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? ?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*'
_output_shapes
:????????? ?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
dense_1/MatMulMatMuldropout/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^embedding/embedding_lookup?^text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup2?
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
F
__inference__creator_79714
identity: ??MutableHashTable|
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	table_7*
value_dtype0	]
IdentityIdentityMutableHashTable:table_handle:0^NoOp*
T0*
_output_shapes
: Y
NoOpNoOp^MutableHashTable*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2$
MutableHashTableMutableHashTable
?

?
*__inference_sequential_layer_call_fn_79060

inputs
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?N 
	unknown_4:  
	unknown_5: 
	unknown_6: 
	unknown_7:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
	*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_77892o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
D__inference_embedding_layer_call_and_return_conditional_losses_77846

inputs	)
embedding_lookup_77840:	?N 
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_77840inputs*
Tindices0	*)
_class
loc:@embedding_lookup/77840*+
_output_shapes
:?????????d *
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/77840*+
_output_shapes
:?????????d ?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????d w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:?????????d Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????d: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
,
__inference__destroyer_79724
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_79660

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:????????? [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:????????? "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
Y
text_vectorization_input=
*serving_default_text_vectorization_input:0?????????=
dense_12
StatefulPartitionedCall_1:0?????????tensorflow/serving/predict:??
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
P
	keras_api
_lookup_layer
_adapt_function"
_tf_keras_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

embeddings"
_tf_keras_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
?
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

&kernel
'bias"
_tf_keras_layer
?
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses
._random_generator"
_tf_keras_layer
?
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses

5kernel
6bias"
_tf_keras_layer
C
1
&2
'3
54
65"
trackable_list_wrapper
C
0
&1
'2
53
64"
trackable_list_wrapper
 "
trackable_list_wrapper
?
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
<trace_0
=trace_1
>trace_2
?trace_32?
*__inference_sequential_layer_call_fn_77913
*__inference_sequential_layer_call_fn_79060
*__inference_sequential_layer_call_fn_79083
*__inference_sequential_layer_call_fn_78281?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z<trace_0z=trace_1z>trace_2z?trace_3
?
@trace_0
Atrace_1
Btrace_2
Ctrace_32?
E__inference_sequential_layer_call_and_return_conditional_losses_79337
E__inference_sequential_layer_call_and_return_conditional_losses_79598
E__inference_sequential_layer_call_and_return_conditional_losses_78529
E__inference_sequential_layer_call_and_return_conditional_losses_78777?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z@trace_0zAtrace_1zBtrace_2zCtrace_3
?B?
 __inference__wrapped_model_77588text_vectorization_input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?
Diter

Ebeta_1

Fbeta_2
	Gdecay
Hlearning_ratem?&m?'m?5m?6m?v?&v?'v?5v?6v?"
	optimizer
,
Iserving_default"
signature_map
"
_generic_user_object
L
J	keras_api
Klookup_table
Ltoken_counts"
_tf_keras_layer
?
Mtrace_02?
__inference_adapt_step_79037?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zMtrace_0
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
Strace_02?
)__inference_embedding_layer_call_fn_79605?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zStrace_0
?
Ttrace_02?
D__inference_embedding_layer_call_and_return_conditional_losses_79614?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zTtrace_0
':%	?N 2embedding/embeddings
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
Ztrace_02?
8__inference_global_average_pooling1d_layer_call_fn_79619?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zZtrace_0
?
[trace_02?
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_79625?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z[trace_0
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
?
atrace_02?
%__inference_dense_layer_call_fn_79634?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zatrace_0
?
btrace_02?
@__inference_dense_layer_call_and_return_conditional_losses_79645?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zbtrace_0
:  2dense/kernel
: 2
dense/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
?
htrace_0
itrace_12?
'__inference_dropout_layer_call_fn_79650
'__inference_dropout_layer_call_fn_79655?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 zhtrace_0zitrace_1
?
jtrace_0
ktrace_12?
B__inference_dropout_layer_call_and_return_conditional_losses_79660
B__inference_dropout_layer_call_and_return_conditional_losses_79672?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 zjtrace_0zktrace_1
"
_generic_user_object
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
?
lnon_trainable_variables

mlayers
nmetrics
olayer_regularization_losses
player_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
?
qtrace_02?
'__inference_dense_1_layer_call_fn_79681?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zqtrace_0
?
rtrace_02?
B__inference_dense_1_layer_call_and_return_conditional_losses_79691?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zrtrace_0
 : 2dense_1/kernel
:2dense_1/bias
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
*__inference_sequential_layer_call_fn_77913text_vectorization_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
*__inference_sequential_layer_call_fn_79060inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
*__inference_sequential_layer_call_fn_79083inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
*__inference_sequential_layer_call_fn_78281text_vectorization_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
E__inference_sequential_layer_call_and_return_conditional_losses_79337inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
E__inference_sequential_layer_call_and_return_conditional_losses_79598inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
E__inference_sequential_layer_call_and_return_conditional_losses_78529text_vectorization_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
E__inference_sequential_layer_call_and_return_conditional_losses_78777text_vectorization_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
?B?
#__inference_signature_wrapper_78808text_vectorization_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
"
_generic_user_object
f
u_initializer
v_create_resource
w_initialize
x_destroy_resourceR jtf.StaticHashTable
L
y_create_resource
z_initialize
{_destroy_resourceR Z

 ??
?B?
__inference_adapt_step_79037iterator"?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
)__inference_embedding_layer_call_fn_79605inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_embedding_layer_call_and_return_conditional_losses_79614inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
8__inference_global_average_pooling1d_layer_call_fn_79619inputs"?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_79625inputs"?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
%__inference_dense_layer_call_fn_79634inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
@__inference_dense_layer_call_and_return_conditional_losses_79645inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
'__inference_dropout_layer_call_fn_79650inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
'__inference_dropout_layer_call_fn_79655inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
B__inference_dropout_layer_call_and_return_conditional_losses_79660inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
B__inference_dropout_layer_call_and_return_conditional_losses_79672inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
'__inference_dense_1_layer_call_fn_79681inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_dense_1_layer_call_and_return_conditional_losses_79691inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
N
|	variables
}	keras_api
	~total
	count"
_tf_keras_metric
c
?	variables
?	keras_api

?total

?count
?
_fn_kwargs"
_tf_keras_metric
"
_generic_user_object
?
?trace_02?
__inference__creator_79696?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__initializer_79704?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__destroyer_79709?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__creator_79714?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__initializer_79719?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__destroyer_79724?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
.
~0
1"
trackable_list_wrapper
-
|	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
?B?
__inference__creator_79696"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__initializer_79704"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__destroyer_79709"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__creator_79714"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__initializer_79719"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__destroyer_79724"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
,:*	?N 2Adam/embedding/embeddings/m
#:!  2Adam/dense/kernel/m
: 2Adam/dense/bias/m
%:# 2Adam/dense_1/kernel/m
:2Adam/dense_1/bias/m
,:*	?N 2Adam/embedding/embeddings/v
#:!  2Adam/dense/kernel/v
: 2Adam/dense/bias/v
%:# 2Adam/dense_1/kernel/v
:2Adam/dense_1/bias/v
?B?
__inference_save_fn_79743checkpoint_key"?
???
FullArgSpec
args?
jcheckpoint_key
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?	
? 
?B?
__inference_restore_fn_79751restored_tensors_0restored_tensors_1"?
???
FullArgSpec
args? 
varargsjrestored_tensors
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
	?	
J
Constjtf.TrackableConstant
!J	
Const_1jtf.TrackableConstant
!J	
Const_2jtf.TrackableConstant
!J	
Const_3jtf.TrackableConstant
!J	
Const_4jtf.TrackableConstant
!J	
Const_5jtf.TrackableConstant6
__inference__creator_79696?

? 
? "? 6
__inference__creator_79714?

? 
? "? 8
__inference__destroyer_79709?

? 
? "? 8
__inference__destroyer_79724?

? 
? "? A
__inference__initializer_79704K???

? 
? "? :
__inference__initializer_79719?

? 
? "? ?
 __inference__wrapped_model_77588?K???&'56=?:
3?0
.?+
text_vectorization_input?????????
? "1?.
,
dense_1!?
dense_1?????????j
__inference_adapt_step_79037JL???<
5?2
0?-?
??????????IteratorSpec 
? "
 ?
B__inference_dense_1_layer_call_and_return_conditional_losses_79691\56/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? z
'__inference_dense_1_layer_call_fn_79681O56/?,
%?"
 ?
inputs????????? 
? "???????????
@__inference_dense_layer_call_and_return_conditional_losses_79645\&'/?,
%?"
 ?
inputs????????? 
? "%?"
?
0????????? 
? x
%__inference_dense_layer_call_fn_79634O&'/?,
%?"
 ?
inputs????????? 
? "?????????? ?
B__inference_dropout_layer_call_and_return_conditional_losses_79660\3?0
)?&
 ?
inputs????????? 
p 
? "%?"
?
0????????? 
? ?
B__inference_dropout_layer_call_and_return_conditional_losses_79672\3?0
)?&
 ?
inputs????????? 
p
? "%?"
?
0????????? 
? z
'__inference_dropout_layer_call_fn_79650O3?0
)?&
 ?
inputs????????? 
p 
? "?????????? z
'__inference_dropout_layer_call_fn_79655O3?0
)?&
 ?
inputs????????? 
p
? "?????????? ?
D__inference_embedding_layer_call_and_return_conditional_losses_79614_/?,
%?"
 ?
inputs?????????d	
? ")?&
?
0?????????d 
? 
)__inference_embedding_layer_call_fn_79605R/?,
%?"
 ?
inputs?????????d	
? "??????????d ?
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_79625{I?F
??<
6?3
inputs'???????????????????????????

 
? ".?+
$?!
0??????????????????
? ?
8__inference_global_average_pooling1d_layer_call_fn_79619nI?F
??<
6?3
inputs'???????????????????????????

 
? "!???????????????????y
__inference_restore_fn_79751YLK?H
A?>
?
restored_tensors_0
?
restored_tensors_1	
? "? ?
__inference_save_fn_79743?L&?#
?
?
checkpoint_key 
? "???
`?]

name?
0/name 
#

slice_spec?
0/slice_spec 

tensor?
0/tensor
`?]

name?
1/name 
#

slice_spec?
1/slice_spec 

tensor?
1/tensor	?
E__inference_sequential_layer_call_and_return_conditional_losses_78529|K???&'56E?B
;?8
.?+
text_vectorization_input?????????
p 

 
? "%?"
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_78777|K???&'56E?B
;?8
.?+
text_vectorization_input?????????
p

 
? "%?"
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_79337jK???&'563?0
)?&
?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_79598jK???&'563?0
)?&
?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
*__inference_sequential_layer_call_fn_77913oK???&'56E?B
;?8
.?+
text_vectorization_input?????????
p 

 
? "???????????
*__inference_sequential_layer_call_fn_78281oK???&'56E?B
;?8
.?+
text_vectorization_input?????????
p

 
? "???????????
*__inference_sequential_layer_call_fn_79060]K???&'563?0
)?&
?
inputs?????????
p 

 
? "???????????
*__inference_sequential_layer_call_fn_79083]K???&'563?0
)?&
?
inputs?????????
p

 
? "???????????
#__inference_signature_wrapper_78808?K???&'56Y?V
? 
O?L
J
text_vectorization_input.?+
text_vectorization_input?????????"1?.
,
dense_1!?
dense_1?????????